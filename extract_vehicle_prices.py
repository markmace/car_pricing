from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import numpy as np
import time
import re
import os
from datetime import datetime, timedelta
from config import CAR_CONFIG, URL_TEMPLATE, STORAGE_CONFIG


class CarListingScraper:
    def __init__(self, config=CAR_CONFIG, storage_config=STORAGE_CONFIG, force_refresh=False):
        self.config = config
        self.storage_config = storage_config
        self.force_refresh = force_refresh

    def _clean_mileage(self, mileage_text):
        """Convert mileage text to number, handling 'New' cases"""
        if not mileage_text or 'new' in mileage_text.lower():
            return 0
        numbers = re.findall(r'\d+', mileage_text)
        return int(''.join(numbers)) if numbers else None

    def _get_latest_file(self):
        """Find the most recent data file"""
        base_dir = self.storage_config['base_dir']
        if not os.path.exists(base_dir):
            return None

        files = [f for f in os.listdir(base_dir) if f.endswith('.csv')]
        if not files:
            return None

        files.sort(reverse=True)
        return os.path.join(base_dir, files[0])

    def _is_data_fresh(self):
        """Check if we have fresh data within max_age_days"""
        latest_file = self._get_latest_file()
        if not latest_file:
            return False

        file_date_str = os.path.basename(latest_file).split('_')[0]
        file_date = datetime.strptime(file_date_str, '%Y%m%d')
        max_age = timedelta(days=self.storage_config['max_age_days'])

        return datetime.now() - file_date < max_age

    def scrape_listings(self):
        """Main scraping function"""
        if not self.force_refresh and self._is_data_fresh():
            print("Using cached data from today...")
            return pd.read_csv(self._get_latest_file())

        url = URL_TEMPLATE.format(**self.config)

        options = webdriver.ChromeOptions()
        options.add_argument('--headless')
        driver = webdriver.Chrome(options=options)

        try:
            print(f"Searching for {self.config['make'].title()} {self.config['model'].title()} listings...")
            driver.get(url)

            wait = WebDriverWait(driver, 10)
            listings = wait.until(EC.presence_of_all_elements_located(
                (By.CLASS_NAME, "vehicle-card")))

            total_listings = len(listings)
            print(f"Found {total_listings} listings to process...")

            results = []
            for idx, listing in enumerate(listings, 1):
                try:
                    # Basic info
                    title = listing.find_element(By.CLASS_NAME, "title").text
                    price = listing.find_element(By.CLASS_NAME, "primary-price").text
                    dealer = listing.find_element(By.CLASS_NAME, "dealer-name").text

                    # Get year from title
                    year_match = re.search(r'\b(20\d{2})\b', title)
                    year = year_match.group(1) if year_match else "Year not found"

                    # Get mileage
                    try:
                        mileage_element = listing.find_element(By.CLASS_NAME, "mileage")
                        mileage = self._clean_mileage(mileage_element.text)
                    except:
                        mileage = None

                    # Get URL and visit individual listing page
                    url_element = listing.find_element(By.CSS_SELECTOR, "a.vehicle-card-link")
                    listing_url = url_element.get_attribute('href')

                    # Get distance and location
                    try:
                        distance_element = listing.find_element(By.CLASS_NAME, "miles-from")
                        distance = distance_element.text
                        # try to split distance and location, ex:  Lunenburg, MA (35 mi.)
                        location = distance.split(' (')[0].strip()
                        distance = int(distance.split(' (')[1].strip('mi.)')) if re.search(r'\d+', distance) else 0
                    except:
                        distance = "Distance not found"
                        location = "Location not found"

                    # Visit the individual car page
                    # print(f"Visiting detailed page: {listing_url}")
                    driver.execute_script('window.open("","_blank");')  # Open new tab
                    driver.switch_to.window(driver.window_handles[-1])
                    driver.get(listing_url)

                    # Get listing text including the detailed page content
                    listing_text = ""
                    try:
                        # Wait for features section to load
                        detail_wait = WebDriverWait(driver, 5)
                        features_section = detail_wait.until(EC.presence_of_element_located(
                            (By.CLASS_NAME, "features-section")))
                        listing_text += features_section.text.lower() + " "
                    except Exception as e:
                        print(f"Warning: Could not get features for {title}")

                    # Close detail page tab and switch back to main window
                    driver.close()
                    driver.switch_to.window(driver.window_handles[0])

                    # Add main listing text as backup
                    listing_text += listing.text.lower() + " "

                    features = {}
                    # Check each feature and its synonyms
                    for feature_name, synonyms in self.config['features'].items():
                        has_feature = any(synonym.lower() in listing_text for synonym in synonyms)
                        features[f'has_{feature_name}'] = has_feature

                    result_dict = {
                        'year': year,
                        'title': title,
                        'price': price,
                        'mileage': mileage,
                        'dealer': dealer,
                        'location': location,
                        'distance (mi)': distance,
                        'url': listing_url,
                        **features  # Unpack the features dict
                    }

                    results.append(result_dict)
                    print(f"Processed {idx}/{total_listings}: {title}")

                except Exception as e:
                    print(f"Error on listing {idx}/{total_listings}: {str(e)}")
                    continue

            df = pd.DataFrame(results)
            self._save_results(df)
            return df

        finally:
            driver.quit()

    def _save_results(self, df):
        """Save results with timestamp"""
        # Clean up price column with handling for 'Not Priced'
        df['price'] = df['price'].apply(lambda x: np.nan if x == 'Not Priced' else
        float(x.replace('$', '').replace(',', '')))

        # Create timestamp and filename
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{self.config['make']}_{self.config['model']}_listings.csv"

        # Save file
        os.makedirs(self.storage_config['base_dir'], exist_ok=True)
        save_path = os.path.join(self.storage_config['base_dir'], filename)
        df.to_csv(save_path, index=False)
        print(f"\nSaved {len(df)} listings to: {save_path}")


if __name__ == "__main__":
    force_refresh = False
    # You can set force_refresh=True to ignore cached data
    scraper = CarListingScraper(force_refresh=force_refresh)
    self = scraper
    df = scraper.scrape_listings()

    print(df.drop(columns='mileage').sort_values(by='price').to_string(index=False))

    # Import visualization libraries
    import seaborn as sns
    import matplotlib.pyplot as plt

    # Set the style
    sns.set_theme()
    sns.set_palette("husl")

    # Create figure and grid of subplots
    years = sorted(df['year'].unique())
    fig = plt.figure(figsize=(12, 4 * len(years)))

    # Calculate global min and max prices for consistent bins
    all_prices = df['price'].dropna()
    global_min_price = np.floor(all_prices.min() / 500) * 500  # Round down to nearest $500
    global_max_price = np.ceil(all_prices.max() / 500) * 500  # Round up to nearest $500
    bins = np.arange(global_min_price, global_max_price + 500, 500)  # $500 increments

    for idx, year in enumerate(years, 1):
        year_data = df[df['year'] == year]
        ax = plt.subplot(len(years), 1, idx)

        # Create histogram with consistent bins
        hist = sns.histplot(data=year_data,
                            x='price',
                            hue='has_moonroof',
                            multiple="layer",
                            alpha=0.6,
                            bins=bins,
                            stat='count')

        # Customize the plot
        plt.title(f'{year} Subaru Impreza Price Distribution by Moonroof', pad=20)
        plt.xlabel('Price ($)')
        plt.ylabel('Count')

        # Set consistent x-axis limits
        plt.xlim(global_min_price, global_max_price)

        # Format x-axis with bin ranges
        bin_centers = (bins[:-1] + bins[1:]) / 2
        bin_labels = [f'${bins[i]:,.0f}-${bins[i + 1]:,.0f}' for i in range(len(bins) - 1)]

        plt.xticks(bin_centers, bin_labels, rotation=45, ha='right')

        # Update legend and add grid
        plt.legend(title='Has Moonroof', labels=['Yes', 'No'],
                   loc='upper right', fontsize='small')  # Smaller legend font
        plt.grid(True, alpha=0.3)

        # Add statistical summary box for each group
        for i, has_moonroof in enumerate([False, True]):
            subset = year_data[year_data['has_moonroof'] == has_moonroof]
            if not subset.empty:
                prices = subset['price'].dropna()
                n = len(prices)
                sem = prices.std() / np.sqrt(n)  # Calculate Standard Error of Mean

                stats_text = (
                    f"{'No' if not has_moonroof else 'Has'} Moonroof (n={n})\n"
                    f"μ = ${prices.mean():,.0f} ± ${sem:,.0f}\n"
                    f"Median = ${prices.median():,.0f}\n"
                    f"Range: ${prices.min():,.0f}-${prices.max():,.0f}"
                )

                # Position boxes on left side, stacked vertically with more spacing
                y_pos = plt.ylim()[1] * (0.9 - i * 0.3)  # Increased vertical spacing
                x_pos = global_min_price + (global_max_price - global_min_price) * 0.02

                plt.text(x_pos, y_pos,
                         stats_text,
                         bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray',
                                   boxstyle='round,pad=0.5'),
                         ha='left', va='center',
                         fontfamily='monospace',
                         fontsize='x-small')  # Smaller font size
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    # Save the plot
    plot_path = os.path.join(scraper.storage_config['base_dir'],
                             f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_price_distribution.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved price distribution plot to: {plot_path}")

    # Show text summary
    print("\nSummary Statistics by Year and Moonroof:")

    # Get summary statistics grouped by year and moonroof
    summary = df.groupby(['year', 'has_moonroof']).agg({
        'price': ['count', 'mean', lambda x: x.std() / np.sqrt(len(x))]  # Add SEM calculation
    }).round(2)

    # First unstack the moonroof level
    summary = summary.unstack()

    # Rename the columns more carefully
    summary.columns = [
        f'{"Count" if col[0] == "count" else "Mean ($)" if col[0] == "mean" else "SEM ($)"} - {"No" if not col[1] else "Has"} Moonroof'
        for col in summary.columns
    ]

    # Print combined summary
    print("\nCounts and Average Prices (± SEM):")
    print(summary.to_string())
