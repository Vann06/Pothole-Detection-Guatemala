from pathlib import Path
from collections import Counter, defaultdict
import xml.etree.ElementTree as ET
import time

from tqdm import tqdm


# ============================================================
# CONFIGURACIÓN DEL PROYECTO
# ============================================================

# validate_dataset.py está en:
# proyecto/src/data/validate_dataset.py
#
# parents[2] nos lleva hasta:
# proyecto/
PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "RDD2022"


# Dominios que esperamos encontrar en RDD2022.
# Esto también evita que carpetas de respaldo como
# Japan_INCOMPLETE entren accidentalmente al análisis.
EXPECTED_DOMAINS = [
    "China_Drone",
    "China_MotorBike",
    "Czech",
    "India",
    "Japan",
    "Norway",
    "United_States",
]


# Aunque actualmente RDD2022 usa JPG,
# dejamos soporte para otros formatos por robustez.
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
}


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def is_train_path(path: Path) -> bool:
    """
    Devuelve True si el archivo está dentro de una carpeta train.

    Funciona tanto con:

        Japan/train/...

    como con:

        Japan/Japan/train/...
    """
    return "train" in [part.lower() for part in path.parts]


def is_test_path(path: Path) -> bool:
    """
    Devuelve True si el archivo pertenece a test.
    """
    return "test" in [part.lower() for part in path.parts]


def find_images(domain_root: Path, split: str):
    """
    Encuentra imágenes de train o test recursivamente.
    """

    images = []

    for path in domain_root.rglob("*"):

        if not path.is_file():
            continue

        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        if split == "train" and is_train_path(path):
            images.append(path)

        elif split == "test" and is_test_path(path):
            images.append(path)

    return images


def find_xml(domain_root: Path):
    """
    Encuentra únicamente XML que pertenecen a train.
    """

    xml_files = []

    for path in domain_root.rglob("*.xml"):

        if is_train_path(path):
            xml_files.append(path)

    return xml_files


def inspect_xml(xml_path: Path):
    """
    Lee un XML Pascal VOC y devuelve todas las clases
    anotadas en la imagen.

    Ejemplo:
        ["D00", "D20", "D40"]
    """

    tree = ET.parse(xml_path)
    root = tree.getroot()

    classes = []

    for obj in root.findall("object"):

        name = obj.findtext("name")

        if name:
            classes.append(name.strip())

    # Liberamos memoria
    root.clear()

    return classes


def print_section(title):
    """
    Imprime títulos para hacer la salida más clara.
    """

    print()
    print("=" * 75)
    print(title)
    print("=" * 75)


# ============================================================
# VALIDACIÓN PRINCIPAL
# ============================================================

def main():

    start_time = time.time()

    print_section("RDD2022 DATASET VALIDATION")

    print(f"Proyecto: {PROJECT_ROOT}")
    print(f"Dataset:  {DATASET_ROOT}")

    # --------------------------------------------------------
    # 1. Verificar que exista RDD2022
    # --------------------------------------------------------

    if not DATASET_ROOT.exists():

        raise FileNotFoundError(
            f"\nNo se encontró el dataset en:\n{DATASET_ROOT}"
        )

    # --------------------------------------------------------
    # 2. Verificar dominios
    # --------------------------------------------------------

    print_section("1. DOMINIOS ENCONTRADOS")

    existing_domains = []

    for domain in EXPECTED_DOMAINS:

        domain_path = DATASET_ROOT / domain

        if domain_path.exists():

            existing_domains.append(domain)

            print(f"[OK] {domain}")

        else:

            print(f"[FALTA] {domain}")

    if not existing_domains:

        raise RuntimeError(
            "No se encontró ninguno de los dominios esperados."
        )

    # --------------------------------------------------------
    # 3. Encontrar imágenes y XML
    # --------------------------------------------------------

    print_section("2. CONTEO DE ARCHIVOS")

    train_images_by_domain = {}
    test_images_by_domain = {}
    xml_by_domain = {}

    for domain in existing_domains:

        domain_root = DATASET_ROOT / domain

        train_images = find_images(
            domain_root,
            split="train",
        )

        test_images = find_images(
            domain_root,
            split="test",
        )

        xml_files = find_xml(domain_root)

        train_images_by_domain[domain] = train_images
        test_images_by_domain[domain] = test_images
        xml_by_domain[domain] = xml_files

    print(
        f"{'Dominio':20s}"
        f"{'Train IMG':>12s}"
        f"{'Train XML':>12s}"
        f"{'Test IMG':>12s}"
    )

    print("-" * 58)

    total_train_images = 0
    total_train_xml = 0
    total_test_images = 0

    for domain in existing_domains:

        train_count = len(
            train_images_by_domain[domain]
        )

        xml_count = len(
            xml_by_domain[domain]
        )

        test_count = len(
            test_images_by_domain[domain]
        )

        total_train_images += train_count
        total_train_xml += xml_count
        total_test_images += test_count

        print(
            f"{domain:20s}"
            f"{train_count:12d}"
            f"{xml_count:12d}"
            f"{test_count:12d}"
        )

    print("-" * 58)

    print(
        f"{'TOTAL':20s}"
        f"{total_train_images:12d}"
        f"{total_train_xml:12d}"
        f"{total_test_images:12d}"
    )

    print(
        f"\nTotal imágenes RDD2022: "
        f"{total_train_images + total_test_images}"
    )

    # --------------------------------------------------------
    # 4. Correspondencia imagen <-> XML
    # --------------------------------------------------------

    print_section("3. CORRESPONDENCIA IMAGEN <-> XML")

    total_missing_xml = 0
    total_missing_image = 0
    total_duplicate_images = 0
    total_duplicate_xml = 0

    for domain in existing_domains:

        image_stems = [
            path.stem
            for path in train_images_by_domain[domain]
        ]

        xml_stems = [
            path.stem
            for path in xml_by_domain[domain]
        ]

        image_counter = Counter(image_stems)
        xml_counter = Counter(xml_stems)

        image_names = set(image_stems)
        xml_names = set(xml_stems)

        missing_xml = image_names - xml_names
        missing_image = xml_names - image_names

        duplicate_images = [
            name
            for name, count in image_counter.items()
            if count > 1
        ]

        duplicate_xml = [
            name
            for name, count in xml_counter.items()
            if count > 1
        ]

        total_missing_xml += len(missing_xml)
        total_missing_image += len(missing_image)
        total_duplicate_images += len(duplicate_images)
        total_duplicate_xml += len(duplicate_xml)

        print(
            f"{domain:20s}"
            f" sin XML={len(missing_xml):5d}"
            f" | XML sin IMG={len(missing_image):5d}"
            f" | IMG duplicadas={len(duplicate_images):4d}"
            f" | XML duplicados={len(duplicate_xml):4d}"
        )

        # Mostramos algunos ejemplos si existen errores
        if missing_xml:

            print(
                "   Ejemplos IMG sin XML:",
                sorted(missing_xml)[:5],
            )

        if missing_image:

            print(
                "   Ejemplos XML sin IMG:",
                sorted(missing_image)[:5],
            )

    # --------------------------------------------------------
    # 5. Procesar anotaciones
    # --------------------------------------------------------

    print_section("4. PROCESANDO CLASES")

    all_xml = []

    for domain in existing_domains:

        for xml_path in xml_by_domain[domain]:

            all_xml.append(
                (domain, xml_path)
            )

    object_class_counter = Counter()

    object_classes_by_domain = defaultdict(
        Counter
    )

    image_class_counter = Counter()

    image_classes_by_domain = defaultdict(
        Counter
    )

    images_with_d40_by_domain = Counter()

    images_without_d40_by_domain = Counter()

    empty_annotations_by_domain = Counter()

    images_with_any_damage_by_domain = Counter()

    corrupt_xml = []

    for domain, xml_path in tqdm(
        all_xml,
        desc="Leyendo XML",
        unit="xml",
    ):

        try:

            classes = inspect_xml(xml_path)

            # ---------------------------------------------
            # Conteo de objetos
            # ---------------------------------------------

            object_class_counter.update(classes)

            object_classes_by_domain[domain].update(
                classes
            )

            # ---------------------------------------------
            # Conteo a nivel de imagen
            # ---------------------------------------------

            unique_classes = set(classes)

            image_class_counter.update(
                unique_classes
            )

            image_classes_by_domain[domain].update(
                unique_classes
            )

            # ---------------------------------------------
            # D40 / potholes
            # ---------------------------------------------

            if "D40" in unique_classes:

                images_with_d40_by_domain[
                    domain
                ] += 1

            else:

                images_without_d40_by_domain[
                    domain
                ] += 1

            # ---------------------------------------------
            # Imágenes sin objetos anotados
            # ---------------------------------------------

            if len(classes) == 0:

                empty_annotations_by_domain[
                    domain
                ] += 1

            else:

                images_with_any_damage_by_domain[
                    domain
                ] += 1

        except Exception as exc:

            corrupt_xml.append(
                (
                    xml_path,
                    str(exc),
                )
            )

    # --------------------------------------------------------
    # 6. Clases encontradas
    # --------------------------------------------------------

    print_section("5. OBJETOS POR CLASE")

    labels = sorted(
        object_class_counter.keys()
    )

    if not labels:

        print("No se encontraron clases.")

    else:

        header = f"{'Dominio':20s}"

        for label in labels:

            header += f"{label:>10s}"

        print(header)
        print("-" * len(header))

        for domain in existing_domains:

            row = f"{domain:20s}"

            for label in labels:

                row += (
                    f"{object_classes_by_domain[domain][label]:10d}"
                )

            print(row)

        print("-" * len(header))

        total_row = f"{'TOTAL OBJETOS':20s}"

        for label in labels:

            total_row += (
                f"{object_class_counter[label]:10d}"
            )

        print(total_row)

    # --------------------------------------------------------
    # 7. Imágenes que contienen cada clase
    # --------------------------------------------------------

    print_section("6. IMÁGENES QUE CONTIENEN CADA CLASE")

    if labels:

        header = f"{'Dominio':20s}"

        for label in labels:

            header += f"{label:>10s}"

        print(header)
        print("-" * len(header))

        for domain in existing_domains:

            row = f"{domain:20s}"

            for label in labels:

                row += (
                    f"{image_classes_by_domain[domain][label]:10d}"
                )

            print(row)

    # --------------------------------------------------------
    # 8. Resumen para potholes
    # --------------------------------------------------------

    print_section("7. RESUMEN D40 / POTHOLES")

    print(
        f"{'Dominio':20s}"
        f"{'Con D40':>12s}"
        f"{'Sin D40':>12s}"
        f"{'Sin objetos':>14s}"
        f"{'Con daño':>12s}"
    )

    print("-" * 70)

    total_with_d40 = 0
    total_without_d40 = 0
    total_empty = 0
    total_damage = 0

    for domain in existing_domains:

        with_d40 = images_with_d40_by_domain[
            domain
        ]

        without_d40 = images_without_d40_by_domain[
            domain
        ]

        empty = empty_annotations_by_domain[
            domain
        ]

        damage = images_with_any_damage_by_domain[
            domain
        ]

        total_with_d40 += with_d40
        total_without_d40 += without_d40
        total_empty += empty
        total_damage += damage

        print(
            f"{domain:20s}"
            f"{with_d40:12d}"
            f"{without_d40:12d}"
            f"{empty:14d}"
            f"{damage:12d}"
        )

    print("-" * 70)

    print(
        f"{'TOTAL':20s}"
        f"{total_with_d40:12d}"
        f"{total_without_d40:12d}"
        f"{total_empty:14d}"
        f"{total_damage:12d}"
    )

    # --------------------------------------------------------
    # 9. Errores encontrados
    # --------------------------------------------------------

    print_section("8. CONTROL DE CALIDAD")

    print(
        f"Train imágenes sin XML:       "
        f"{total_missing_xml}"
    )

    print(
        f"XML sin imagen correspondiente:"
        f" {total_missing_image}"
    )

    print(
        f"Imágenes con nombre duplicado: "
        f"{total_duplicate_images}"
    )

    print(
        f"XML con nombre duplicado:      "
        f"{total_duplicate_xml}"
    )

    print(
        f"XML corruptos:                 "
        f"{len(corrupt_xml)}"
    )

    if corrupt_xml:

        print(
            "\nPrimeros XML con problemas:"
        )

        for path, error in corrupt_xml[:10]:

            print(f"\n{path}")
            print(f"   ERROR: {error}")

    # --------------------------------------------------------
    # 10. Resultado final
    # --------------------------------------------------------

    print_section("9. RESULTADO FINAL")

    structural_ok = (
        len(existing_domains) == len(EXPECTED_DOMAINS)
        and total_missing_xml == 0
        and total_missing_image == 0
        and len(corrupt_xml) == 0
        and total_duplicate_images == 0
        and total_duplicate_xml == 0
    )

    if structural_ok:

        print(
            "DATASET VALIDADO CORRECTAMENTE"
        )

        print(
            "No se encontraron problemas "
            "estructurales."
        )

    else:

        print(
            "SE ENCONTRARON PROBLEMAS "
            "QUE DEBEN REVISARSE."
        )

    elapsed = time.time() - start_time

    print(
        f"\nTiempo total: {elapsed:.2f} segundos"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()
