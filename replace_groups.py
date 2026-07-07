import argparse
import zipfile

BATCHES = {
    1: range(1, 26),
    2: range(26, 51),
    3: range(51, 76),
    4: range(76, 101),
    5: range(101, 126),
    6: range(126, 151),
    7: range(151, 176),
    8: range(176, 201),
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--groups", required=True, help="Example: 1,2,3,4")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    groups = [int(x.strip()) for x in args.groups.split(",") if x.strip()]
    replace_names = set()

    for g in groups:
        for idx in BATCHES[g]:
            replace_names.add(f"{idx}.png")

    with zipfile.ZipFile(args.base, "r") as base_zip, \
         zipfile.ZipFile(args.variant, "r") as var_zip, \
         zipfile.ZipFile(args.out, "w", zipfile.ZIP_DEFLATED) as out_zip:

        for idx in range(1, 201):
            name = f"{idx}.png"
            src = var_zip if name in replace_names else base_zip
            out_zip.writestr(name, src.read(name))

    print(f"Saved {args.out}; replaced WM groups {groups}")

if __name__ == "__main__":
    main()