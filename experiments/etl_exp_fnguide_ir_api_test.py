"""
Experiment 14: Test FnGuide IR API with curl_cffi

Goal: Test API access and examine data structure from FnGuide IR events endpoint.

API Pattern:
- URL: https://comp.fnguide.com/SVO2/json/data/05_01/YYYYMM.json?_={timestamp}
- Example: https://comp.fnguide.com/SVO2/json/data/05_01/202508.json?_=1761466294746

Expected response:
{
  "comp": [
    {
      "KEY": "A14108020250626000035",
      "일련번호": "20250626000035",
      "기준일자": "01",
      "기업명": "리가켐바이오",
      ...
    }
  ]
}

This experiment will:
1. Test a single API request
2. Print the response structure
3. Analyze field names and data types
4. Determine if pagination/rate limiting is needed
"""

import json
import time
from curl_cffi import requests
from datetime import datetime


def print_section(title: str):
    """Print a formatted section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def test_single_request(year: int, month: int):
    """Test a single API request for given year-month.

    Args:
        year: Year (e.g., 2025)
        month: Month (1-12)
    """
    # Format: YYYYMM
    year_month = f"{year:04d}{month:02d}"

    # Current timestamp (milliseconds since epoch)
    timestamp = int(time.time() * 1000)

    # Build URL
    url = f"https://comp.fnguide.com/SVO2/json/data/05_01/{year_month}.json?_={timestamp}"

    print(f"\n[Request]")
    print(f"  URL: {url}")
    print(f"  Year-Month: {year_month}")

    try:
        # Use curl_cffi with Chrome impersonation
        response = requests.get(url, impersonate="chrome")

        print(f"\n[Response Status]")
        print(f"  Status Code: {response.status_code}")
        print(f"  Content-Type: {response.headers.get('Content-Type', 'N/A')}")
        print(f"  Content-Length: {len(response.content)} bytes")

        if response.status_code == 200:
            # Parse JSON - handle EUC-KR encoding for Korean
            try:
                data = response.json()
            except UnicodeDecodeError:
                # Try decoding with EUC-KR (Korean encoding)
                content_text = response.content.decode('euc-kr')
                data = json.loads(content_text)

            print(f"\n[JSON Structure]")
            print(f"  Top-level keys: {list(data.keys())}")

            if 'comp' in data and len(data['comp']) > 0:
                records = data['comp']
                print(f"  Total records: {len(records)}")

                # Analyze first record
                first_record = records[0]
                print(f"\n[First Record]")
                print(f"  Keys: {list(first_record.keys())}")
                print(f"\n  Full record:")
                for key, value in first_record.items():
                    print(f"    {key}: {value} (type: {type(value).__name__})")

                # Show sample of all records (first 5)
                print(f"\n[Sample Records (first 5)]")
                for i, record in enumerate(records[:5], 1):
                    print(f"\n  Record {i}:")
                    # Show key fields only for brevity
                    print(f"    기업명: {record.get('기업명', 'N/A')}")
                    print(f"    이벤트코드: {record.get('이벤트코드', 'N/A')}")
                    print(f"    이벤트명: {record.get('이벤트명', 'N/A')}")
                    print(f"    기준일자: {record.get('기준일자', 'N/A')}")
                    print(f"    일련번호: {record.get('일련번호', 'N/A')}")

                # Analyze all field names across all records
                print(f"\n[All Field Names Across Records]")
                all_keys = set()
                for record in records:
                    all_keys.update(record.keys())

                print(f"  Total unique fields: {len(all_keys)}")
                print(f"  Fields:")
                for key in sorted(all_keys):
                    print(f"    - {key}")

                # Check data types for each field
                print(f"\n[Field Data Types (from first record)]")
                for key, value in first_record.items():
                    print(f"    {key}: {type(value).__name__} (sample: {repr(value)[:50]})")

                # Print first record as JSON with proper encoding
                print(f"\n[First Record as JSON (properly encoded)]")
                print(json.dumps(first_record, ensure_ascii=False, indent=2))

                return data
            else:
                print(f"\n  [WARNING] No 'comp' data or empty array")
                return data
        else:
            print(f"\n  [ERROR] Request failed with status {response.status_code}")
            print(f"  Response text (first 500 chars): {response.text[:500]}")
            return None

    except Exception as e:
        print(f"\n  [ERROR] {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    print_section("FnGuide IR API Test Experiment")

    print("\n[Test Plan]")
    print("  1. Test recent month (2025-08)")
    print("  2. Test old month (2010-01)")
    print("  3. Test month with likely no data (future)")

    # Test 1: Recent month (August 2025)
    print_section("Test 1: Recent Month (2025-08)")
    data_recent = test_single_request(2025, 8)

    # Test 2: Old month (January 2010)
    print_section("Test 2: Old Month (2010-01)")
    data_old = test_single_request(2010, 1)

    # Test 3: Current month (to check if it returns data)
    current_year = datetime.now().year
    current_month = datetime.now().month
    print_section(f"Test 3: Current Month ({current_year}-{current_month:02d})")
    data_current = test_single_request(current_year, current_month)

    # Summary
    print_section("Summary")

    print("\n[Results]")
    print(f"  Recent month (2025-08): {'[OK] Success' if data_recent else '[FAIL] Failed'}")
    print(f"  Old month (2010-01): {'[OK] Success' if data_old else '[FAIL] Failed'}")
    print(f"  Current month ({current_year}-{current_month:02d}): {'[OK] Success' if data_current else '[FAIL] Failed'}")

    print("\n[Findings]")
    if data_recent:
        print(f"  - API is accessible")
        print(f"  - Returns JSON with 'comp' array")
        print(f"  - curl_cffi works for this endpoint")

    print("\n[Next Steps]")
    print("  1. Determine date range to scrape (2010-01 to 2025-09)")
    print("  2. Identify all field names and types")
    print("  3. Design schema for Parquet storage")
    print("  4. Handle potential errors (empty months, 404s)")
    print("  5. Add rate limiting (be respectful to server)")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
