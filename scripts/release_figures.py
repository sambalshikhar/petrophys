"""Extract fresh notebook figures and contact sheets for release review."""
from __future__ import annotations

import argparse
import base64
import io
import hashlib
import json
from pathlib import Path


def main():
    from PIL import Image, ImageDraw, ImageFont

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-root", type=Path, default=Path("outputs/release_audit"))
    parser.add_argument("--notebooks-dir", type=Path, help="Prefer fresh executions from this directory")
    args = parser.parse_args()
    root = args.audit_root.resolve()
    directory = root / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    selected = {}
    index_path = root / "evidence" / "figure-index.json"
    previous = {row["ID"]: row for row in json.loads(index_path.read_text(encoding="utf8"))} if index_path.exists() else {}
    # Prefer outputs executed with the complete optional-dependency environment.
    folders = [root / "notebooks", root / "notebooks-full311"]
    if args.notebooks_dir is not None:
        folders.append(args.notebooks_dir.resolve())
    for folder in folders:
        for path in sorted(folder.rglob("*.ipynb")):
            selected[path.relative_to(folder).as_posix()] = path
    rows, thumbnails = [], []
    for name, path in sorted(selected.items()):
        notebook = json.loads(path.read_text(encoding="utf8"))
        for i, cell in enumerate(notebook["cells"]):
            for j, output in enumerate(cell.get("outputs", [])):
                png = output.get("data", {}).get("image/png")
                if not png:
                    continue
                data = base64.b64decode("".join(png) if isinstance(png, list) else png, validate=True)
                number = len(rows) + 1
                filename = f"{number:03d}-{Path(name).stem}-cell{i}.png"
                target = directory / filename
                target.write_bytes(data)
                image = Image.open(io.BytesIO(data)).convert("RGB")
                rows.append({"ID": f"{name}:{cell.get('id', i)}:{j}", "Notebook": name,
                             "Cell": i, "Cell ID": cell.get("id"), "Image": str(target),
                             "Width": image.width, "Height": image.height, "Status": "PENDING",
                             "SHA256": hashlib.sha256(data).hexdigest(),
                             "Execution source": str(path), "Review notes": "Readable image decode only; scientific/visual review pending."})
                old = previous.get(rows[-1]["ID"], {})
                if old.get("SHA256") == rows[-1]["SHA256"]:
                    for field in ("Status", "Reviewer", "Review decision", "Review notes", "Contact-sheet review", "Full-resolution review"):
                        if field in old:
                            rows[-1][field] = old[field]
                elif old:
                    rows[-1]["Review notes"] = "Image changed; previous review no longer applies. " + rows[-1]["Review notes"]
                image.thumbnail((560, 380))
                thumbnails.append((number, name, i, image.copy()))
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    for start in range(0, len(thumbnails), 8):
        sheet = Image.new("RGB", (2320, 900), "#e7ebef")
        draw = ImageDraw.Draw(sheet)
        for offset, (number, name, cell, image) in enumerate(thumbnails[start:start + 8]):
            x, y = (offset % 4) * 580, (offset // 4) * 450
            draw.text((x + 8, y + 6), f"{number:03d}: {Path(name).stem[:53]} / cell {cell}", font=font, fill="black")
            sheet.paste(image, (x + 8, y + 42))
        sheet.save(directory / f"contact-{start // 8 + 1:02d}.png")
    index_path.write_text(json.dumps(rows, indent=2), encoding="utf8")
    print(f"Extracted {len(rows)} fresh figures; {((len(rows) + 7) // 8)} contact sheets. Visual review remains pending.")


if __name__ == "__main__":
    main()
