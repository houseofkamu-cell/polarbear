"""Golden Hour LUT + cinematic sunset plate.
LUT: 33pt .cube, Rec.709 / Gamma 2.4 in and out (display-referred).
"""
import numpy as np
from PIL import Image, ImageFilter
import os, sys
OUT = os.path.dirname(os.path.abspath(__file__))
rng = np.random.default_rng(7)

# ---------- colour helpers ----------
LUMA = np.array([0.2126, 0.7152, 0.0722])
def luma(x): return x @ LUMA
def smooth(e0, e1, x):
    t = np.clip((x - e0) / (e1 - e0), 0, 1); return t * t * (3 - 2 * t)

def rgb_to_hsv(c):
    r, g, b = c[..., 0], c[..., 1], c[..., 2]
    mx, mn = c.max(-1), c.min(-1); d = mx - mn
    h = np.zeros_like(mx); m = d > 1e-6
    rc = np.where(m, (mx - r) / np.where(m, d, 1), 0)
    gc = np.where(m, (mx - g) / np.where(m, d, 1), 0)
    bc = np.where(m, (mx - b) / np.where(m, d, 1), 0)
    h = np.where(r == mx, bc - gc, np.where(g == mx, 2 + rc - bc, 4 + gc - rc))
    h = np.where(m, (h / 6) % 1, 0)
    s = np.where(mx > 1e-6, d / np.where(mx > 1e-6, mx, 1), 0)
    return np.stack([h, s, mx], -1)

def hsv_to_rgb(c):
    h, s, v = c[..., 0] % 1, c[..., 1], c[..., 2]
    i = np.floor(h * 6).astype(int) % 6; f = h * 6 - np.floor(h * 6)
    p, q, t = v * (1 - s), v * (1 - s * f), v * (1 - s * (1 - f))
    sel = [np.stack(a, -1) for a in [(v,t,p),(q,v,p),(p,v,t),(p,q,v),(t,p,v),(v,p,q)]]
    out = np.zeros(c.shape)
    for k in range(6): out = np.where((i == k)[..., None], sel[k], out)
    return out

# ---------- the grade ----------
def grade(rgb):
    x = np.clip(rgb, 0, 1).astype(np.float64)

    # 1. Hue-selective work before the look (like HSL qualifiers in a node tree)
    hsv = rgb_to_hsv(x)
    h, s = hsv[..., 0], hsv[..., 1]
    def band(c, w):  # soft hue window, hue in 0..1
        d = np.abs(((h - c + 0.5) % 1) - 0.5); return smooth(w, 0, d) * smooth(0.05, 0.25, s)
    skin = band(0.07, 0.06)                       # ~25 deg: protect skin
    blue = band(0.60, 0.12); green = band(0.30, 0.10)
    h = h + blue * (-0.035) + green * (-0.025)     # blues -> teal, greens -> olive/amber
    s = s * (1 - 0.22 * blue - 0.30 * green) * (1 + 0.04 * skin)
    x = hsv_to_rgb(np.stack([h, np.clip(s, 0, 1), hsv[..., 2]], -1))

    # 2. Tone curve: filmic S, lifted toe, soft shoulder
    Y = luma(x)
    toe, shoulder = 0.028, 0.965
    curve = lambda v: toe + (shoulder - toe) * (
        0.5 + 0.5 * np.tanh(2.35 * (v - 0.47)) / np.tanh(2.35 * 0.53 * 1.0))
    Yc = np.clip(curve(Y), 0, 1)
    lo = curve(0.0); hi = curve(1.0)
    Yc = toe + (Yc - lo) / (hi - lo) * (shoulder - toe)
    ratio = np.where(Y > 1e-5, Yc / np.maximum(Y, 1e-5), 1)[..., None]
    x = x * ratio
    x = np.where((Y < 1e-5)[..., None], Yc[..., None], x)

    # 3. Split tone in luma ranges (offset-style, sums to near-neutral mid grey)
    Y = luma(np.clip(x, 0, 1))[..., None]
    w_sh = smooth(0.45, 0.0, Y); w_hi = smooth(0.45, 1.0, Y)
    w_mid = np.clip(1 - w_sh - w_hi, 0, 1)
    shadow = np.array([-0.006, 0.004, 0.022]) * 1.0      # deep cool plum-teal for depth
    shadow += np.array([0.010, -0.004, 0.006])           # pull towards plum, not pure teal
    mids   = np.array([0.032, 0.010, -0.026])            # amber
    highs  = np.array([0.030, 0.016, -0.026])            # gold/peach
    x = x + w_sh * shadow + w_mid * mids + w_hi * highs

    # 4. White balance warmth on the whole image (gain), then protect neutral white
    x = x * np.array([1.035, 1.000, 0.950])

    # 5. Luminance-dependent saturation: rich mids, soft highlights, quiet blacks
    Y = luma(np.clip(x, 0, 1))[..., None]
    sat = 1.08 - 0.30 * smooth(0.65, 1.0, Y) - 0.25 * smooth(0.12, 0.0, Y)
    x = Y + (x - Y) * sat

    # 6. Highlight rolloff into warm cream (stops hard clipping of sun)
    Y = luma(np.clip(x, 0, 1))[..., None]
    hi = smooth(0.70, 1.0, Y) * 0.6
    cream = np.array([1.0, 0.94, 0.83]); cream = cream / (cream @ LUMA)
    x = x * (1 - hi) + cream * Y * hi

    # 7. Soft gamut compress
    x = np.where(x > 0.9, 0.9 + 0.1 * np.tanh((x - 0.9) / 0.1), x)
    x = np.where(x < 0.0, 0.0, x)
    return np.clip(x, 0, 1)

# ---------- write LUT ----------
N = 33
g = np.linspace(0, 1, N)
b_, g_, r_ = np.meshgrid(g, g, g, indexing="ij")         # .cube: red varies fastest
cube_in = np.stack([r_, g_, b_], -1).reshape(-1, 3)
cube_out = grade(cube_in)
with open(os.path.join(OUT, "GoldenHour_Rec709_33.cube"), "w") as f:
    f.write('TITLE "Golden Hour v1"\n# Input/Output: Rec.709 Gamma 2.4. For log footage, CST to Rec.709 first.\n')
    f.write(f"LUT_3D_SIZE {N}\nDOMAIN_MIN 0.0 0.0 0.0\nDOMAIN_MAX 1.0 1.0 1.0\n")
    for v in cube_out: f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
LUT = cube_out.reshape(N, N, N, 3)  # [b,g,r]

def apply_lut(img):
    p = np.clip(img, 0, 1) * (N - 1)
    i0 = np.floor(p).astype(int).clip(0, N - 2); f = p - i0
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    o = 0
    for db in (0, 1):
        for dg in (0, 1):
            for dr in (0, 1):
                w = (fr if dr else 1 - fr) * (fg if dg else 1 - fg) * (fb if db else 1 - fb)
                o = o + w * LUT[b0 + db, g0 + dg, r0 + dr]
    return o

# ---------- procedural plate ----------
def fbm(h, w, octaves=6, base=4, seed=0, stretch=1.0):
    r = np.random.default_rng(seed); acc = np.zeros((h, w)); amp = 1; tot = 0
    for o in range(octaves):
        gh, gw = base * 2**o, int(base * 2**o * w / h / stretch) + 1
        n = Image.fromarray((r.random((gh + 1, gw + 1)) * 255).astype(np.uint8))
        acc += amp * np.asarray(n.resize((w, h), Image.BICUBIC), np.float64) / 255
        tot += amp; amp *= 0.5
    return acc / tot

def plate(W, H):
    y = np.linspace(0, 1, H)[:, None]; xx = np.linspace(0, 1, W)[None, :]
    hz = 0.64                                         # horizon line
    sx, sy = 0.66, hz - 0.035                         # sun just above horizon
    # sky gradient (top dusk violet -> rose -> amber -> hot band at horizon)
    stops = [(0.00, (0.10, 0.09, 0.20)), (0.22, (0.27, 0.16, 0.28)), (0.42, (0.62, 0.30, 0.27)),
             (0.56, (0.90, 0.52, 0.28)), (hz, (1.00, 0.74, 0.40))]
    sky = np.zeros((H, 1, 3))
    for (a, ca), (b, cb) in zip(stops, stops[1:]):
        t = np.clip((y - a) / (b - a), 0, 1)[..., None]
        m = ((y >= a) & (y <= b))[..., None]
        sky = np.where(m, np.array(ca) * (1 - t) + np.array(cb) * t, sky)
    sky = np.broadcast_to(sky, (H, W, 3)).copy()
    ar = W / H
    d = np.sqrt(((xx - sx) * ar) ** 2 + (y - sy) ** 2)
    glow = np.exp(-d / 0.09)[..., None] * np.array([1.0, 0.62, 0.30]) * 0.55
    halo = np.exp(-d / 0.32)[..., None] * np.array([0.9, 0.45, 0.25]) * 0.30
    sky = sky + glow + halo
    # clouds: stretched fbm, lit from below by sun
    n = fbm(H, W, 7, 6, seed=3, stretch=4.0)
    n = np.asarray(Image.fromarray((n * 255).astype(np.uint8)).resize((W, H)), np.float64) / 255
    band = smooth(0.08, 0.30, y) * smooth(hz - 0.02, 0.38, y)
    cl = smooth(0.50, 0.72, n) * band
    rim = np.exp(-d / 0.25)
    cloud_col = (np.array([0.32, 0.17, 0.25]) * (1 - rim[..., None]) +
                 np.array([1.0, 0.58, 0.32]) * rim[..., None])
    sky = sky * (1 - 0.8 * cl[..., None]) + cloud_col * 0.8 * cl[..., None]
    # sun disc
    disc = smooth(0.026, 0.020, d)[..., None]
    sky = sky * (1 - disc) + np.array([1.0, 0.96, 0.85]) * 1.3 * disc
    img = sky
    # water below horizon with sun glitter path
    wy = np.clip((y - hz) / (1 - hz), 0, 1)
    below = (y > hz)
    refl = np.flipud(np.clip(sky, 0, 1.4))                    # mirrored sky approx
    ry = (2 * hz - y).clip(0, 1); idx = (ry * (H - 1)).astype(int)[:, 0]
    refl = sky[idx]
    ripple = fbm(H, W, 5, 60, seed=11, stretch=12.0)
    ripple = np.asarray(Image.fromarray((ripple * 255).astype(np.uint8)).resize((W, H)), np.float64) / 255
    path = np.exp(-np.abs((xx - sx) * ar) / (0.03 + 0.25 * wy)) * smooth(0.55, 0.85, ripple)
    water = refl * (0.55 - 0.35 * wy)[..., None] + np.array([1.0, 0.70, 0.40]) * (path * 0.9)[..., None]
    water = water + np.array([0.05, 0.04, 0.09]) * wy[..., None]
    img = np.where(below[..., None], water, img)
    # headland silhouettes, two layers with atmospheric perspective
    for k, (base, amp, col, haze) in enumerate([(hz - 0.03, 0.06, (0.33, 0.18, 0.22), 0.55),
                                                 (hz + 0.005, 0.10, (0.08, 0.05, 0.08), 0.0)]):
        prof = fbm(1, W, 6, 2, seed=20 + k)[0]
        mask_x = smooth(0.55 + 0.12 * k, 0.0, xx[0])        # land on the left
        top = base - amp * prof * mask_x - 0.002
        m = ((y > top[None, :]) & (y <= hz + 0.002 * k))[..., None]
        c = np.array(col) * (1 - haze) + np.array([0.95, 0.55, 0.35]) * 0.5 * haze
        img = np.where(m, c, img)
    # atmospheric haze at horizon
    hzb = np.exp(-np.abs(y - hz) / 0.035)[..., None] * np.array([1.0, 0.68, 0.42]) * 0.22
    img = img + hzb
    # bloom
    u8 = Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8))
    bright = np.clip(img - 0.75, 0, None)
    bl = Image.fromarray((np.clip(bright * 2, 0, 1) * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(W / 60))
    img = img + np.asarray(bl, np.float64) / 255 * np.array([0.9, 0.55, 0.30]) * 0.6
    # vignette
    v = ((xx - 0.5) * ar / ar) ** 2 * 1.2 + (y - 0.5) ** 2 * 1.6
    img = img * (1 - 0.45 * smooth(0.05, 0.55, v))[..., None]
    return np.clip(img, 0, 1)

def add_grain(img, strength=0.022, seed=1):
    r = np.random.default_rng(seed)
    gr = r.normal(0, 1, img.shape[:2])
    gr = np.asarray(Image.fromarray(((gr * 40 + 128).clip(0, 255)).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(0.7)), np.float64) / 255 - 0.5
    Y = luma(img)[..., None]
    w = 4 * Y * (1 - Y) + 0.15                                # grain strongest in mids
    return np.clip(img + gr[..., None] * strength * 2 * w, 0, 1)

def save(a, name):
    Image.fromarray((np.clip(a, 0, 1) * 255 + 0.5).astype(np.uint8)).save(os.path.join(OUT, name), quality=95)

W, H = (3840, 2160) if "--draft" not in sys.argv else (1280, 720)
raw = plate(W, H)
graded = add_grain(apply_lut(raw))
save(graded, "GoldenHour_plate_UHD.png")
small = lambda a: np.asarray(Image.fromarray((a * 255).astype(np.uint8)).resize((1280, 720), Image.LANCZOS), np.float64) / 255
save(np.concatenate([small(raw), small(graded)], 1), "preview_before_after.jpg")

# test chart: grey ramp + hue sweep + skin patch, before/after
cw, ch = 1280, 360
ramp = np.tile(np.linspace(0, 1, cw)[None, :, None], (120, 1, 3))
hues = np.linspace(0, 1, cw, endpoint=False)
sweep = hsv_to_rgb(np.stack([np.tile(hues, (120, 1)), np.full((120, cw), 0.75), np.full((120, cw), 0.8)], -1))
skins = np.concatenate([np.tile(np.array(c)[None, None], (120, cw // 4, 1)) for c in
                        [(0.95, 0.80, 0.70), (0.80, 0.60, 0.48), (0.55, 0.38, 0.28), (0.36, 0.24, 0.18)]], 1)
chart = np.concatenate([ramp, sweep, skins], 0)
save(np.concatenate([chart, apply_lut(chart)], 0), "chart_before_after.png")
print("done", W, H)
