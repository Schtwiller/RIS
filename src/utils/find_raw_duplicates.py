from pathlib import Path

RAW = Path("../../data/raw")
to_delete = []
all_files = list(RAW.glob("*.*"))
print(len(all_files))

for img in RAW.glob("*.*"):
    name_lower = img.name.lower()
    if (
        " copy" in name_lower
        or "copy (" in name_lower
        or name_lower.endswith("copy.jpg")
        or "copy" in name_lower.split("_")[-1]
    ):
        to_delete.append(img)

print(f"🧹 Found {len(to_delete)} files to delete.")
confirm = input("Delete these files? [y/N]: ").lower()

if confirm == "y":
    for f in to_delete:
        try:
            f.unlink()
            print(f"🗑️  Deleted: {f.name}")
        except Exception as e:
            print(f"❌ Could not delete {f.name}: {e}")
else:
    print("❎ Aborted. No files deleted.")
