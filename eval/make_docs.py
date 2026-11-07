"""Regenerate the PDFs in eval/docs from the page text below (stdlib only).

The documents are fictional on purpose, so a model cannot answer from general
knowledge and every question has exactly one right source page.

    python eval/make_docs.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

OUT = Path(__file__).parent / "docs"

DOCS: dict[str, list[tuple[str, str]]] = {
    "harbor_light_ferry.pdf": [
        (
            "Tickets and fares",
            "The adult single fare on the Harbor Light ferry is 14 euros. Children under 12 "
            "travel for half price, and infants under 3 travel free. A ten-trip pass costs "
            "110 euros and expires 90 days after the first sailing. Tickets bought on board "
            "cost 2 euros more than tickets bought at the Pennick Quay booking desk.",
        ),
        (
            "Timetable",
            "The first sailing leaves Pennick Quay at 06:40 and the last sailing leaves at "
            "21:15. The crossing to Grey Island takes 35 minutes. The winter timetable, from "
            "November to March, runs one sailing every hour. The summer timetable runs one "
            "sailing every 30 minutes. There is no service on Christmas Day.",
        ),
        (
            "Vehicles, bicycles and luggage",
            "Cars up to 5.0 metres long are carried for 38 euros. Bicycles travel free in "
            "the rear rack. Each passenger may bring two bags of up to 20 kg each at no "
            "charge. Dogs must be kept on a lead on the open deck. Assistance animals travel "
            "free and may sit anywhere on board.",
        ),
        (
            "Cancellations and safety",
            "Sailings are cancelled when the wind exceeds Beaufort force 8. Refunds for "
            "cancelled sailings are issued within 14 days through the booking desk. "
            "Lifejackets are stored under the seats on the lower deck. The ferry MV Cormorant "
            "carries up to 220 passengers.",
        ),
    ],
    "orchard_street_garden.pdf": [
        (
            "Membership",
            "The annual plot fee at the Orchard Street Community Garden is 45 euros. Each "
            "member receives one plot of 12 square metres. Members must volunteer for 6 hours "
            "each year on shared tasks. The waiting list for a plot is currently about "
            "8 months.",
        ),
        (
            "Garden rules",
            "Gardening is allowed from 07:00 to 20:00 every day. Chemical pesticides are "
            "banned, and only organic compost may be used. Bonfires are prohibited. Water "
            "comes from the shared rain tanks, and hoses must never be left running "
            "unattended.",
        ),
        (
            "Shared facilities",
            "The tool shed is unlocked on Saturdays from 10:00 to 14:00 by the key warden. "
            "The shed code changes every March. The Parks Department empties the compost bins "
            "once a month. The garden beehives are looked after by Ines Okafor.",
        ),
    ],
    "vela7_weather_station.pdf": [
        (
            "Overview",
            "The Vela-7 weather station records temperature, humidity, air pressure and wind "
            "speed. By default it takes one reading every 10 minutes. It is powered by a "
            "20-watt solar panel that charges a 12 Ah battery.",
        ),
        (
            "Installation",
            "Mount the Vela-7 mast 2 metres above the ground. Keep the station at least "
            "3 metres away from walls and trees. Secure the mast with four guy wires. Use the "
            "built-in bubble indicator to level the unit.",
        ),
        (
            "Calibration",
            "Calibrate the temperature sensor every 12 months in an ice bath at 0 degrees "
            "Celsius. The pressure sensor is calibrated at the factory and drifts by at most "
            "0.5 hPa per year. The tipping-bucket rain gauge measures 0.2 mm of rain per tip.",
        ),
        (
            "Troubleshooting and data",
            "A blinking red LED means the battery is low. A steady blue LED means the station "
            "is uploading data. Hold the reset button for 10 seconds to restart the unit. "
            "Readings are stored as CSV files on a microSD card of up to 32 GB.",
        ),
    ],
}


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _page_stream(title: str, body: str) -> bytes:
    ops = ["BT", "/F1 16 Tf", "72 740 Td", f"({_escape(title)}) Tj", "ET"]
    ops += ["BT", "/F1 11 Tf", "14 TL", "72 700 Td"]
    for line in textwrap.wrap(body, 80):
        ops.append(f"({_escape(line)}) Tj T*")
    ops.append("ET")
    return "\n".join(ops).encode("latin-1")


def build_pdf(pages: list[tuple[str, str]]) -> bytes:
    objs: list[bytes] = []  # object n is objs[n - 1]
    n = len(pages)
    kids = " ".join(f"{3 + 2 * i} 0 R" for i in range(n))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {n} >>".encode())
    font_id = 3 + 2 * n
    for i, (title, body) in enumerate(pages):
        content_id = 4 + 2 * i
        objs.append(
            (
                "<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                f"/Contents {content_id} 0 R /Resources << /Font << /F1 {font_id} 0 R >> >> >>"
            ).encode()
        )
        stream = _page_stream(title, body)
        objs.append(f"<< /Length {len(stream)} >>\nstream\n".encode() + stream + b"\nendstream")
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")

    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for i, obj in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + obj + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, pages in DOCS.items():
        (OUT / name).write_bytes(build_pdf(pages))
        print(f"wrote {OUT / name} ({len(pages)} pages)")


if __name__ == "__main__":
    main()
