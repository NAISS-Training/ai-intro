import xml.etree.ElementTree as ET

SVG_NS = "http://www.w3.org/2000/svg"
ET.register_namespace("", SVG_NS)


def animate_small_vs_big(
    svg_path="small_vs_big.svg",
    output_path="small_vs_big_animated.svg",
):
    tree = ET.parse(svg_path)
    root = tree.getroot()

    ns = {"svg": SVG_NS}

    duration = 30
    steps = 25
    io_wait_fraction = 0.1

    x_shift = 150

    # Small file animations
    # 0: Start
    # t1: envelope1 arrive to meet-up, file1 move start
    # t2: file1 ariving, both start moving
    # t2+dt: envelope2 moving
    # t3: envelope2 arriving, file2 start moving
    # t4: file2 arriving, both start moving
    # t4+dt: envelope3 start moving
    # ...

    # Envelopes
    # 1: 0, t1, t2
    # 2: t2+dt, t3, t4
    # 3: t4+dt, t5, t6
    # ...
    small_envelope_gids = [
        "g63020",
        "g63020-6",
        "g63020-5",
        "g63020-8",
        "g63020-7",
        "g63020-2",
        "g63020-61",
        "g63020-9",
        "g63020-71",
        "g63020-97",
        "g63020-3",
    ]
    for i_gid, gid in enumerate(small_envelope_gids):
        group = root.find(f".//svg:g[@id='{gid}']", ns)
        if group is None:
            raise RuntimeError(f"Didn't find group by ID {gid}")

        key_steps = [0, 1, 2] + [
            2 * (i_envelope + 1) + t_sub
            for i_envelope in range(i_gid)
            for t_sub in [io_wait_fraction, 1, 2]
        ] + [2 * i_gid + 3, steps]
        print(i_gid, key_steps)
        x_values = [0 if (i / 3) < (i_gid + 1) else x_shift for i in range(len(key_steps))]
        y_values = [0] + [
            20 + (3 + 2 / 3) * int(i // 3) for i in range(len(key_steps) - 1)
        ]
        animate = ET.Element(
            "animateTransform",
            {
                "attributeName": "transform",
                "additive": "sum",
                "type": "translate",
                "dur": f"{duration}s",
                "fill": "freeze",
                "keyTimes": ";".join(f"{step / steps}" for step in key_steps),
                "values": ";".join([f"{x} {y}" for x, y in zip(x_values, y_values)]),
                "repeatCount": "indefinite",
            },
        )

        group.append(animate)

    # Files
    # 0: 0
    # 1: t1, t2, t3
    # 2: t3, t4, t4
    # 3: t5, t6, t7
    # ...
    small_file_ids_pos: list[tuple[str, tuple[float, float]]] = [
        ("g30423", (47., -30.)),
        ("g35633", (30.5, -63.)),
        ("g35638", (-21.5, -63.)),
        ("g35648", (12., -33.)),
        ("g35671", (47., -32.5)),
        ("g35676", (-24., -33.)),
        ("g55540", (47., -62.)),
        ("g55545", (12., -63.)),
        ("g55550", (-5., -32.)),
        ("g55555", (-5., -62.)),
        ("g55560", (30., -32.3)),
    ]
    for i_gid, (gid, (x_pos, y_pos)) in enumerate(small_file_ids_pos):
        group = root.find(f".//svg:g[@id='{gid}']", ns)
        if group is None:
            raise RuntimeError(f"Didn't find group by ID {gid}")

        step_size = 1 / steps
        central_time = (2 * i_gid + 2) * step_size
        key_times = [0, central_time - step_size, central_time, central_time + step_size, 1]

        x_values = [0, 0, x_pos, x_pos + x_shift, x_pos + x_shift]
        y_values = [0, 0, y_pos, y_pos, y_pos]

        animate = ET.Element(
            "animateTransform",
            {
                "attributeName": "transform",
                "additive": "sum",
                "type": "translate",
                "dur": f"{duration}s",
                "fill": "freeze",
                "keyTimes": ";".join(f"{t}" for t in key_times),
                "values": ";".join([f"{x} {y}" for x, y in zip(x_values, y_values)]),
                "repeatCount": "indefinite",
            },
        )

        group.append(animate)

    # Large file animations

    # Envelope
    gid = "g63016"
    group = root.find(f".//svg:g[@id='{gid}']", ns)
    if group is None:
        raise RuntimeError(f"Didn't find group by ID {gid}")

    large_repeat_factor = 2
    key_times = [0, large_repeat_factor / steps,  2 * large_repeat_factor / steps, 3 * large_repeat_factor / steps, 1]
    x_values = [0, 0, 0, x_shift, x_shift]
    y_values = [0, 20, 20, 20, 20]
    animate = ET.Element(
        "animateTransform",
        {
            "attributeName": "transform",
            "additive": "sum",
            "type": "translate",
            "dur": f"{duration / large_repeat_factor}s",
            "fill": "freeze",
            "keyTimes": ";".join(f"{t}" for t in key_times),
            "values": ";".join(f"{x} {y}" for x, y in zip(x_values, y_values)),
            "repeatCount": "indefinite",
        },
    )

    group.append(animate)

    # Files
    small_file_ids_pos: list[tuple[str, tuple[float, float]]] = [
        ("g55565", (-10, -30)),  # (0, 2)
        ("g55565-0", (-27.45, -27)),  # (0, 3)
        ("g55565-06", (-44.4, -24)),  # (0, 4)
        ("g55565-1", (8.2, -51.)),  # (1, 1)
        ("g55565-7", (-27.45, -48)),  # (1, 3)
        ("g55565-3", (-10, -45)),  # (1, 2)
        ("g55565-2", (-44.4, -42)),  # (1, 4)
        ("g55565-30", (24.5, -39.2)),  # (1, 0)
        ("g55565-4", (7.3, -6)),  # (0, 1)
    ]
    large_repeat_factor = 2
    key_times = [0, large_repeat_factor / steps,  2 * large_repeat_factor / steps, 3 * large_repeat_factor / steps, 1]
    for i_gid, (gid, (x_pos, y_pos)) in enumerate(small_file_ids_pos):
        group = root.find(f".//svg:g[@id='{gid}']", ns)
        if group is None:
            raise RuntimeError(f"Didn't find group by ID {gid}")

        x_values = [0, 0, x_pos, x_pos + x_shift, x_pos + x_shift]
        y_values = [0, 0, y_pos, y_pos, y_pos]

        animate = ET.Element(
            "animateTransform",
            {
                "attributeName": "transform",
                "additive": "sum",
                "type": "translate",
                "dur": f"{duration / large_repeat_factor}s",
                "fill": "freeze",
                "keyTimes": ";".join(f"{t}" for t in key_times),
                "values": ";".join([f"{x} {y}" for x, y in zip(x_values, y_values)]),
                "repeatCount": "indefinite",
            },
        )

        group.append(animate)

    tree.write(output_path, encoding="utf-8", xml_declaration=True)


if __name__ == "__main__":
    animate_small_vs_big()
