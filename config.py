# Car search configuration
CAR_CONFIG = {
    'make': 'subaru',
    'model': 'impreza',
    'trim': 'subaru-impreza-rs',
    'zip_code': '02144',
    'max_distance': 100,
    'stock_type': 'new',
    'features': {
        'moonroof': [
            'moonroof', 
            'sunroof', 
            'sun roof', 
            'moon roof', 
            'panoramic roof', 
            'glass roof', 
            'sunroof/moonroof', 
            'moonroof/sunroof',
            'power moonroof',
            'power sunroof',
            'pwr moonroof',
            'pwr sunroof'
        ],
    }
}

# Search URL template
URL_TEMPLATE = "https://www.cars.com/shopping/results/?dealer_id=&include_shippable=true&keyword=&list_price_max=&list_price_min=&makes[]={make}&maximum_distance={max_distance}&mileage_max=&models[]={make}-{model}&monthly_payment=&page=1&page_size=100&sort=list_price&stock_type={stock_type}&trims[]={trim}&year_max=&year_min=&zip={zip_code}"

# Data storage configuration
STORAGE_CONFIG = {
    'base_dir': '/Users/mark/car_truth/data',
    'max_age_days': 1  # Maximum age of data before refreshing
} 