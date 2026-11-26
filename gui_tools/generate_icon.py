"""Generate multi-resolution Windows ICO and PNG icon for MACE Trajectory Screener."""
from pathlib import Path
import math
from PIL import Image, ImageDraw, ImageFilter


def create_icon(size: int = 256) -> Image.Image:
    # High-resolution supersampling canvas
    scale = 2
    canvas_size = size * scale
    img = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)

    # 1. Rounded hexagon / squircle base
    pad = 16 * scale
    r = 48 * scale
    x0, y0, x1, y1 = pad, pad, canvas_size - pad, canvas_size - pad

    # Dark high-tech gradient background
    for step in range(canvas_size):
        ratio = step / canvas_size
        r_col = int(15 * (1 - ratio) + 20 * ratio)
        g_col = int(23 * (1 - ratio) + 40 * ratio)
        b_col = int(42 * (1 - ratio) + 85 * ratio)
        draw.line([(0, step), (canvas_size, step)], fill=(r_col, g_col, b_col, 255))

    # Mask to rounded rectangle
    mask = Image.new("L", (canvas_size, canvas_size), 0)
    mask_draw = ImageDraw.Draw(mask)
    mask_draw.rounded_rectangle([pad, pad, canvas_size - pad, canvas_size - pad], radius=r, fill=255)

    base = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    base.paste(img, mask=mask)
    draw = ImageDraw.Draw(base)

    # Draw border highlight
    draw.rounded_rectangle([pad, pad, canvas_size - pad, canvas_size - pad], radius=r, outline=(59, 130, 246, 200), width=4 * scale)

    # 2. Trajectory orbital ellipse (dynamic swoosh)
    cx, cy = canvas_size // 2, canvas_size // 2
    swoosh_box = [cx - 80 * scale, cy - 65 * scale, cx + 80 * scale, cy + 65 * scale]
    draw.arc(swoosh_box, start=-40, end=190, fill=(99, 102, 241, 180), width=3 * scale)

    # 3. Funnel / Screening Cone (subtle geometry in upper-to-center)
    funnel_pts = [
        (cx - 55 * scale, cy - 50 * scale),
        (cx + 55 * scale, cy - 50 * scale),
        (cx + 20 * scale, cy + 10 * scale),
        (cx - 20 * scale, cy + 10 * scale),
    ]
    draw.polygon(funnel_pts, fill=(30, 58, 138, 120), outline=(56, 189, 248, 220), width=2 * scale)

    # Filter grid lines inside funnel
    draw.line([(cx - 30 * scale, cy - 15 * scale), (cx + 30 * scale, cy - 15 * scale)], fill=(56, 189, 248, 140), width=2 * scale)
    draw.line([(cx - 15 * scale, cy + 2 * scale), (cx + 15 * scale, cy + 2 * scale)], fill=(56, 189, 248, 180), width=2 * scale)

    # 4. Filtered focused beam downward (streamlined output structures)
    draw.line([(cx, cy + 12 * scale), (cx, cy + 50 * scale)], fill=(16, 185, 129, 230), width=3 * scale)
    # Downward chevron / arrow
    draw.line([(cx - 10 * scale, cy + 42 * scale), (cx, cy + 52 * scale)], fill=(16, 185, 129, 240), width=3 * scale)
    draw.line([(cx + 10 * scale, cy + 42 * scale), (cx, cy + 52 * scale)], fill=(16, 185, 129, 240), width=3 * scale)

    # 5. Nanocluster / Molecule Atoms (Gold, Copper, Oxygen, Carbon nodes)
    nodes = [
        # Center core (Gold / Au - warm amber)
        (cx - 18 * scale, cy - 35 * scale, 9 * scale, (245, 158, 11, 255), (254, 243, 199, 255)),
        (cx + 15 * scale, cy - 38 * scale, 8 * scale, (217, 119, 6, 255), (253, 230, 138, 255)),
        # Oxygen (red / coral)
        (cx + 35 * scale, cy - 25 * scale, 7 * scale, (239, 68, 68, 255), (254, 202, 202, 255)),
        # Carbon (slate / cyan)
        (cx - 38 * scale, cy - 20 * scale, 7 * scale, (148, 163, 184, 255), (241, 245, 249, 255)),
        # Concentrated output single atom (Emerald / preserved relaxed target)
        (cx, cy + 62 * scale, 10 * scale, (16, 185, 129, 255), (209, 250, 229, 255)),
    ]

    # Bonds between top cluster atoms
    bonds = [
        (nodes[0], nodes[1]),
        (nodes[1], nodes[2]),
        (nodes[0], nodes[3]),
    ]
    for n1, n2 in bonds:
        draw.line([(n1[0], n1[1]), (n2[0], n2[1])], fill=(203, 213, 225, 160), width=2 * scale)

    # Draw atom spheres with highlight glow
    for x, y, rad, fill, highlight in nodes:
        draw.ellipse([x - rad, y - rad, x + rad, y + rad], fill=fill, outline=(255, 255, 255, 180), width=1 * scale)
        # Specular glint
        glint_r = max(1, rad // 3)
        draw.ellipse([x - rad // 2, y - rad // 2, x - rad // 2 + glint_r, y - rad // 2 + glint_r], fill=highlight)

    # Downscale supersampled image smoothly
    final_img = base.resize((size, size), Image.Resampling.LANCZOS)
    return final_img


def main():
    out_dir = Path(__file__).resolve().parent
    icon_256 = create_icon(256)
    png_path = out_dir / "icon.png"
    ico_path = out_dir / "icon.ico"

    icon_256.save(png_path, "PNG")
    print(f"Saved {png_path}")

    # Generate multi-size ICO
    sizes = [(256, 256), (128, 128), (64, 64), (48, 48), (32, 32), (16, 16)]
    icon_256.save(ico_path, format="ICO", sizes=sizes)
    print(f"Saved {ico_path} with sizes: {sizes}")


if __name__ == "__main__":
    main()
