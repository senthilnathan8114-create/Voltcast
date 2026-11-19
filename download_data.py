"""Download the UCI Individual Household Electric Power Consumption dataset."""
from pathlib import Path
import zipfile
import requests

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "data" / "raw"
RAW.mkdir(parents=True, exist_ok=True)
URL = "https://archive.ics.uci.edu/static/public/235/individual+household+electric+power+consumption.zip"
ZIP = RAW / "household_power_consumption.zip"
TARGET = RAW / "household_power_consumption.txt"

if TARGET.exists():
    print(f"Dataset already exists: {TARGET}")
else:
    print("Downloading UCI household power dataset...")
    response = requests.get(URL, timeout=180)
    response.raise_for_status()
    ZIP.write_bytes(response.content)
    with zipfile.ZipFile(ZIP) as archive:
        archive.extractall(RAW)
    matches = list(RAW.rglob("household_power_consumption.txt"))
    if not matches:
        raise FileNotFoundError("household_power_consumption.txt was not found in the UCI archive.")
    if matches[0] != TARGET:
        TARGET.write_bytes(matches[0].read_bytes())
    print(f"Dataset ready: {TARGET}")
