"""All evaluated runs under the merged AIME-2024 convention (I and II are 15 problems each,
so the merged 30-problem score is their mean).  mean-6 = aime24, aime25, amc, beyondaime, hmmt25, hmmt26."""
ROWS = {
"DeepSeek-R1-Distill-1.5B (16k eval)": [
 ("base",                              [35.0,18.3,24.2,62.0, 8.2,15.8,15.9]),
 ("JustRL direct  s50",                [38.3,18.3,26.7,67.5,12.2,19.2,18.2]),
 ("JustRL direct  s100",               [48.3,21.7,22.5,69.6,14.5,19.2,20.5]),
 ("JustRL direct  s200",               [51.7,31.7,34.2,71.1,14.0,20.0,20.2]),
 ("JustRL direct  s250",               [50.0,33.3,30.8,69.0,15.2,19.2,20.5]),
 ("shape only  s50",                   [36.7,23.3,28.3,66.9, 9.5,15.0,17.4]),
 ("shape only  s100",                  [40.0,21.7,23.3,67.2,11.8,14.2,18.9]),
 ("prefs only  s100",                  [43.3,28.3,25.0,68.4,12.8,15.0,18.9]),
 ("R1-7B prefs + JustRL shape  s50",   [41.7,26.7,25.0,61.4, 9.2,15.0,12.9]),
 ("uniform T*=1.287  s50  (broken)",   [36.7,18.3,25.0,63.9,10.8,17.5,18.2]),
 ("uniform T*=1.287  s200 (broken)",   [38.3,23.3,22.5,63.6,12.2,11.7,16.7]),
 ("R1-Distill-7B direct  s200",        [31.7,20.0,22.5,61.7,11.2,15.0,17.4]),
 ("Nemotron-BroRL direct  s200",       [58.3,40.0,35.0,74.7,17.5,21.7,20.5]),
 ("DeepScaleR direct  s200",           [40.7,33.3,30.8,70.5,13.5,15.8,16.7]),
 ("Skywork-OR1-7B direct  s200",       [46.7,25.0,23.3,65.4,12.2,18.3,18.4]),
 ("Light-R1-7B direct  s200",          [35.0,23.3,25.0,63.0, 8.2,15.8,15.9]),
],
"Qwen3-1.7B non-thinking (8k eval)": [
 ("base",                              [20.0, 3.3, 8.3,39.2, 4.2, 3.3,11.4]),
 ("Math-Step500 direct  s50",          [40.0,18.3,25.0,64.2,10.2,17.5,15.9]),
 ("Math-Step500 direct  s200",         [50.0,16.7,23.3,65.1,12.8,14.2,21.2]),
 ("B prefs + A shape  s50",            [41.7,10.0,20.8,62.0, 9.8,14.2,19.7]),
 ("B prefs + student shape  s50",      [38.3,11.7,20.8,57.8, 7.5,13.3,12.9]),
 ("A prefs + B shape  s50",            [23.3, 6.7,11.7,46.7, 5.2, 6.1,16.7]),
 ("A prefs + B shape  s200",           [21.7, 3.3,10.8,47.6, 2.8, 6.7, 9.1]),
 ("A prefs + student shape  s50",      [21.7, 5.0, 9.2,41.6, 4.2, 5.0, 6.8]),
 ("A prefs + uniform T*=0.89  s50",    [18.3, 3.3, 7.5,41.0, 5.5, 5.0, 9.1]),
 ("A = Qwen3-4B direct  s200",         [20.0, 6.7, 8.3,38.9, 5.2, 4.2,11.4]),
 ("Qwen3-8B direct  s200",             [23.3, 6.7,10.0,43.4, 5.0, 9.2, 9.1]),
 ("8B-GRPO-MATH direct  s200",         [26.7, 6.7,13.3,51.5, 7.5, 8.3,11.4]),
 ("4B-Code-1200 direct  s200",         [31.7,13.3,17.5,55.7,10.2, 9.2,15.9]),
]}
H = ["aime24(30)","aime25(30)","amc(83)","beyond(100)","hmmt25(30)","hmmt26(33)"]
for grp, rows in ROWS.items():
    print(f"\n### {grp}")
    print("| run | " + " | ".join(H) + " | mean-6 | Δ |")
    print("|---" * 9 + "|")
    base = None
    for n, v in rows:
        a24 = (v[0] + v[1]) / 2
        cells = [a24, v[2], v[3], v[4], v[5], v[6]]
        m6 = sum(cells) / 6
        if base is None: base = m6
        d = "—" if n == "base" else f"{m6-base:+.1f}"
        print(f"| {n} | " + " | ".join(f"{c:.1f}" for c in cells) + f" | **{m6:.1f}** | {d} |")
