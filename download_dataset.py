"""
Quick dataset downloader for Hillstrom Email Marketing dataset
This tries multiple methods to download the dataset reliably
"""
import pandas as pd
import urllib.request
import ssl

print("="*70)
print("HILLSTROM DATASET DOWNLOADER")
print("="*70)

# Create unverified SSL context (some servers have cert issues)
ssl_context = ssl._create_unverified_context()

# Try multiple sources
sources = [
    {
        "name": "MineThatData (Official)",
        "url": "https://blog.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
    },
    {
        "name": "Archive.org Mirror",
        "url": "https://web.archive.org/web/20200101000000/https://blog.minethatdata.com/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
    },
    {
        "name": "GitHub Mirror 1",
        "url": "https://raw.githubusercontent.com/rmhorton/uplift_modeling_tutorial/master/data/Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv"
    }
]

df = None
for source in sources:
    try:
        print(f"\n📥 Trying: {source['name']}")
        print(f"   URL: {source['url'][:80]}...")
        
        # Download with SSL bypass
        with urllib.request.urlopen(source['url'], context=ssl_context, timeout=30) as response:
            df = pd.read_csv(response)
            
        if df is not None and len(df) > 0:
            print(f"✅ Success! Downloaded {len(df):,} records")
            df.to_csv('hillstrom.csv', index=False)
            print(f"✅ Saved as: hillstrom.csv")
            print(f"\n📊 Dataset Info:")
            print(f"   Rows: {len(df):,}")
            print(f"   Columns: {len(df.columns)}")
            print(f"   Size: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
            break
            
    except Exception as e:
        print(f"❌ Failed: {str(e)[:100]}")
        continue

if df is None:
    print("\n" + "="*70)
    print("❌ ALL DOWNLOAD METHODS FAILED")
    print("="*70)
    print("\n📋 MANUAL DOWNLOAD INSTRUCTIONS:")
    print("\n1. Open browser and visit:")
    print("   https://blog.minethatdata.com/2008/03/minethatdata-e-mail-analytics-and-data.html")
    print("\n2. Click the download link for:")
    print("   Kevin_Hillstrom_MineThatData_E-MailAnalytics_DataMiningChallenge_2008.03.20.csv")
    print("\n3. Save the file and rename it to: hillstrom.csv")
    print("\n4. Place it in this directory:")
    print(f"   C:\\Users\\infam\\ChurnPrevention\\")
    print("\n5. Then run: python hillstrom_analysis.py")
    print("="*70)
else:
    print("\n" + "="*70)
    print("🎉 DATASET READY! Run: python hillstrom_analysis.py")
    print("="*70)
