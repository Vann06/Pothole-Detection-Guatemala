from pathlib import Path
from collections import Counter
import xml.etree.ElementTree as ET
import csv
import time

from tqdm import tqdm


# ============================================================
# CONFIGURACIÓN
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "RDD2022"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "manifests"
)

IMAGES_CSV = OUTPUT_DIR / "images.csv"
ANNOTATIONS_CSV = OUTPUT_DIR / "annotations.csv"


EXPECTED_DOMAINS = [
    "China_Drone",
    "China_MotorBike",
    "Czech",
    "India",
    "Japan",
    "Norway",
    "United_States",
]


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


# Clases principales de RDD2022 que interesan
# directamente al proyecto.
CANONICAL_CLASSES = [
    "D00",
    "D10",
    "D20",
    "D40",
]


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def is_train_path(path: Path) -> bool:
    """
    Indica si un archivo pertenece al split train.
    """

    return "train" in [
        part.lower()
        for part in path.parts
    ]


def find_train_images(domain_root: Path):
    """
    Busca todas las imágenes del train de un dominio.
    """

    images = []

    for path in domain_root.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if not is_train_path(path):
            continue

        images.append(path)

    return images


def find_train_xml(domain_root: Path):
    """
    Busca todos los XML de train.
    """

    return [
        path
        for path in domain_root.rglob("*.xml")
        if is_train_path(path)
    ]


def relative_path(path: Path) -> str:
    """
    Convierte una ruta absoluta en una ruta relativa
    respecto a RDD2022.

    Así no guardamos:
    C:/Projects/...

    sino:
    Japan/train/images/xxx.jpg
    """

    return path.relative_to(
        DATASET_ROOT
    ).as_posix()


def build_image_index(domain_root: Path):
    """
    Crea un diccionario:

    nombre_de_archivo -> Path

    para localizar rápidamente la imagen de cada XML.
    """

    index = {}

    for image_path in find_train_images(domain_root):

        index[image_path.stem] = image_path

    return index


def parse_xml(xml_path: Path):
    """
    Lee un XML Pascal VOC.

    Devuelve:

    - nombre de archivo
    - ancho
    - alto
    - profundidad
    - lista de objetos
    """

    tree = ET.parse(xml_path)

    root = tree.getroot()

    filename = root.findtext("filename")

    size = root.find("size")

    width = None
    height = None
    depth = None

    if size is not None:

        width_text = size.findtext("width")
        height_text = size.findtext("height")
        depth_text = size.findtext("depth")

        if width_text:
            width = int(float(width_text))

        if height_text:
            height = int(float(height_text))

        if depth_text:
            depth = int(float(depth_text))

    objects = []

    for obj in root.findall("object"):

        label = obj.findtext("name")

        if label:
            label = label.strip()

        bbox = obj.find("bndbox")

        if bbox is None:
            continue

        xmin = int(
            float(
                bbox.findtext("xmin", "0")
            )
        )

        ymin = int(
            float(
                bbox.findtext("ymin", "0")
            )
        )

        xmax = int(
            float(
                bbox.findtext("xmax", "0")
            )
        )

        ymax = int(
            float(
                bbox.findtext("ymax", "0")
            )
        )

        bbox_width = max(
            0,
            xmax - xmin,
        )

        bbox_height = max(
            0,
            ymax - ymin,
        )

        bbox_area = (
            bbox_width
            * bbox_height
        )

        objects.append(
            {
                "class": label,
                "xmin": xmin,
                "ymin": ymin,
                "xmax": xmax,
                "ymax": ymax,
                "bbox_width": bbox_width,
                "bbox_height": bbox_height,
                "bbox_area": bbox_area,
            }
        )

    root.clear()

    return {
        "filename": filename,
        "width": width,
        "height": height,
        "depth": depth,
        "objects": objects,
    }


# ============================================================
# CONSTRUCCIÓN DE MANIFESTS
# ============================================================

def main():
    if (OUTPUT_DIR / "dataset_v1.lock.json").exists():
        raise FileExistsError("dataset_v1 is frozen; creating new manifests requires a new dataset version.")

    start_time = time.time()

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 75)
    print("BUILD RDD2022 MANIFEST")
    print("=" * 75)

    image_rows = []
    annotation_rows = []

    global_labels = Counter()

    # ========================================================
    # Procesar dominio por dominio
    # ========================================================

    for domain in EXPECTED_DOMAINS:

        domain_root = (
            DATASET_ROOT
            / domain
        )

        if not domain_root.exists():

            print(
                f"[WARNING] No existe: {domain}"
            )

            continue

        print()
        print("-" * 75)
        print(f"Procesando: {domain}")
        print("-" * 75)

        image_index = build_image_index(
            domain_root
        )

        xml_files = find_train_xml(
            domain_root
        )

        print(
            f"Imágenes: {len(image_index)}"
        )

        print(
            f"XML:      {len(xml_files)}"
        )

        # ====================================================
        # Procesar cada XML
        # ====================================================

        for xml_path in tqdm(
            xml_files,
            desc=domain,
            unit="xml",
        ):

            parsed = parse_xml(
                xml_path
            )

            stem = xml_path.stem

            image_path = image_index.get(
                stem
            )

            if image_path is None:

                raise RuntimeError(
                    "No se encontró imagen para:\n"
                    f"{xml_path}"
                )

            # ------------------------------------------------
            # ID globalmente único
            # ------------------------------------------------

            sample_id = (
                f"{domain}__{stem}"
            )

            objects = parsed["objects"]

            classes = [
                obj["class"]
                for obj in objects
            ]

            class_counts = Counter(
                classes
            )

            global_labels.update(
                classes
            )

            # ------------------------------------------------
            # Flags principales
            # ------------------------------------------------

            has_d00 = int(
                class_counts["D00"] > 0
            )

            has_d10 = int(
                class_counts["D10"] > 0
            )

            has_d20 = int(
                class_counts["D20"] > 0
            )

            has_d40 = int(
                class_counts["D40"] > 0
            )

            has_any_damage = int(
                len(objects) > 0
            )

            is_empty_annotation = int(
                len(objects) == 0
            )

            # Otras etiquetas no pertenecientes
            # a las cuatro clases principales.
            other_labels = sorted(
                {
                    label
                    for label in classes
                    if label
                    not in CANONICAL_CLASSES
                }
            )

            # ------------------------------------------------
            # Una fila por IMAGEN
            # ------------------------------------------------

            image_rows.append(
                {
                    "sample_id": sample_id,
                    "country": domain,
                    "image_path": relative_path(
                        image_path
                    ),
                    "xml_path": relative_path(
                        xml_path
                    ),
                    "width": parsed["width"],
                    "height": parsed["height"],
                    "depth": parsed["depth"],
                    "num_objects": len(
                        objects
                    ),
                    "num_D00": class_counts[
                        "D00"
                    ],
                    "num_D10": class_counts[
                        "D10"
                    ],
                    "num_D20": class_counts[
                        "D20"
                    ],
                    "num_D40": class_counts[
                        "D40"
                    ],
                    "has_D00": has_d00,
                    "has_D10": has_d10,
                    "has_D20": has_d20,
                    "has_D40": has_d40,
                    "has_any_damage":
                        has_any_damage,
                    "is_empty_annotation":
                        is_empty_annotation,
                    "has_other_label": int(
                        len(other_labels) > 0
                    ),
                    "other_labels": ";".join(
                        other_labels
                    ),
                }
            )

            # ------------------------------------------------
            # Una fila por OBJETO
            # ------------------------------------------------

            for object_index, obj in enumerate(
                objects
            ):

                bbox_area_ratio = None

                if (
                    parsed["width"]
                    and parsed["height"]
                ):

                    image_area = (
                        parsed["width"]
                        * parsed["height"]
                    )

                    if image_area > 0:

                        bbox_area_ratio = (
                            obj["bbox_area"]
                            / image_area
                        )

                annotation_rows.append(
                    {
                        "sample_id":
                            sample_id,
                        "country":
                            domain,
                        "object_index":
                            object_index,
                        "class":
                            obj["class"],
                        "xmin":
                            obj["xmin"],
                        "ymin":
                            obj["ymin"],
                        "xmax":
                            obj["xmax"],
                        "ymax":
                            obj["ymax"],
                        "bbox_width":
                            obj[
                                "bbox_width"
                            ],
                        "bbox_height":
                            obj[
                                "bbox_height"
                            ],
                        "bbox_area":
                            obj[
                                "bbox_area"
                            ],
                        "bbox_area_ratio":
                            bbox_area_ratio,
                    }
                )

    # ========================================================
    # GUARDAR images.csv
    # ========================================================

    image_fields = [
        "sample_id",
        "country",
        "image_path",
        "xml_path",
        "width",
        "height",
        "depth",
        "num_objects",
        "num_D00",
        "num_D10",
        "num_D20",
        "num_D40",
        "has_D00",
        "has_D10",
        "has_D20",
        "has_D40",
        "has_any_damage",
        "is_empty_annotation",
        "has_other_label",
        "other_labels",
    ]

    with IMAGES_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=image_fields,
        )

        writer.writeheader()

        writer.writerows(
            image_rows
        )

    # ========================================================
    # GUARDAR annotations.csv
    # ========================================================

    annotation_fields = [
        "sample_id",
        "country",
        "object_index",
        "class",
        "xmin",
        "ymin",
        "xmax",
        "ymax",
        "bbox_width",
        "bbox_height",
        "bbox_area",
        "bbox_area_ratio",
    ]

    with ANNOTATIONS_CSV.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=annotation_fields,
        )

        writer.writeheader()

        writer.writerows(
            annotation_rows
        )

    # ========================================================
    # RESUMEN
    # ========================================================

    elapsed = (
        time.time()
        - start_time
    )

    print()
    print("=" * 75)
    print("MANIFEST GENERADO")
    print("=" * 75)

    print(
        f"Filas images.csv:      "
        f"{len(image_rows)}"
    )

    print(
        f"Filas annotations.csv: "
        f"{len(annotation_rows)}"
    )

    print()
    print("Etiquetas encontradas:")

    for label, count in sorted(
        global_labels.items()
    ):

        print(
            f"  {label:15s} "
            f"{count:8d}"
        )

    print()
    print(
        f"images.csv:\n"
        f"  {IMAGES_CSV}"
    )

    print(
        f"\nannotations.csv:\n"
        f"  {ANNOTATIONS_CSV}"
    )

    print(
        f"\nTiempo total: "
        f"{elapsed:.2f} segundos"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()
