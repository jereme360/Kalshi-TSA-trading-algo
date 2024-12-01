# data/collectors/noaa_collector.py
"""Collector for NOAA weather data at major airports."""

from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple
import pandas as pd
import requests
import time
import logging
from pathlib import Path
from .base_collector import BaseCollector

logger = logging.getLogger(__name__)

class NOAACollector(BaseCollector):
    """Collector for NOAA weather data."""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize NOAA data collector."""
        super().__init__(data_dir)
        self.base_url = "https://api.weather.gov"
        self.data_dir = self.data_dir / "weather"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Major airports with coordinates
        self.airports = {
            'ATL': {'lat': 33.6407, 'lon': -84.4277, 'name': 'Atlanta'},
            'LAX': {'lat': 33.9416, 'lon': -118.4085, 'name': 'Los Angeles'},
            'ORD': {'lat': 41.9742, 'lon': -87.9073, 'name': 'Chicago'},
            'DFW': {'lat': 32.8998, 'lon': -97.0403, 'name': 'Dallas'},
            'DEN': {'lat': 39.8561, 'lon': -104.6737, 'name': 'Denver'},
            'JFK': {'lat': 40.6413, 'lon': -73.7781, 'name': 'New York'},
            'SFO': {'lat': 37.6213, 'lon': -122.3790, 'name': 'San Francisco'},
            'SEA': {'lat': 47.4502, 'lon': -122.3088, 'name': 'Seattle'},
            'LAS': {'lat': 36.0840, 'lon': -115.1537, 'name': 'Las Vegas'},
            'MCO': {'lat': 28.4312, 'lon': -81.3081, 'name': 'Orlando'}
        }
        
        # Cache for station and grid information
        self.station_cache = {}
        self.grid_cache = {}
        
        # Set up session with headers
        self.session = requests.Session()
        self.session.headers.update({
            'Accept': 'application/geo+json',
            'User-Agent': 'TSAPredictionProject/1.0'
        })
    
    def fetch_data(self,
                  start_date: datetime,
                  end_date: Optional[datetime] = None,
                  use_cache: bool = True) -> Dict[str, pd.DataFrame]:
        """
        Fetch weather data for all airports.
        
        Args:
            start_date: Start date for data collection
            end_date: End date for data collection
            use_cache: Whether to use cached data
            
        Returns:
            Dict containing:
                - historical: Historical observations
                - forecast: Current forecasts (if end_date is future)
        """
        end_date = end_date or datetime.now()
        
        try:
            all_data = {}
            
            # Collect data for each airport
            for airport_code in self.airports:
                logger.info(f"Fetching data for {airport_code}")
                
                airport_data = {}
                
                # Get historical observations
                historical = self._fetch_historical_observations(
                    airport_code, start_date, end_date, use_cache
                )
                airport_data['historical'] = historical
                
                # If end_date is in future, get forecast
                if end_date > datetime.now():
                    forecast = self._fetch_forecast(airport_code)
                    airport_data['forecast'] = forecast
                
                all_data[airport_code] = airport_data
                
                # Rate limiting
                time.sleep(0.5)
            
            return all_data
            
        except Exception as e:
            logger.error(f"Error fetching NOAA data: {str(e)}")
            raise
    
    def _get_station_and_grid(self, airport_code: str) -> Tuple[str, Dict]:
        """Get weather station ID and grid points for an airport location."""
        if airport_code in self.station_cache:
            return self.station_cache[airport_code], self.grid_cache[airport_code]
        
        try:
            airport = self.airports[airport_code]
            
            # Get grid points first
            point_url = f"{self.base_url}/points/{airport['lat']},{airport['lon']}"
            response = self.session.get(point_url)
            response.raise_for_status()
            point_data = response.json()
            
            # Extract grid information
            grid_id = point_data['properties']['gridId']
            grid_x = point_data['properties']['gridX']
            grid_y = point_data['properties']['gridY']
            
            # Get nearest station
            stations_url = f"{self.base_url}/points/{airport['lat']},{airport['lon']}/stations"
            response = self.session.get(stations_url)
            response.raise_for_status()
            stations_data = response.json()
            
            # Get closest station ID
            station_id = stations_data['features'][0]['properties']['stationIdentifier']
            
            # Cache the results
            self.station_cache[airport_code] = station_id
            self.grid_cache[airport_code] = {
                'grid_id': grid_id,
                'grid_x': grid_x,
                'grid_y': grid_y
            }
            
            return station_id, self.grid_cache[airport_code]
            
        except Exception as e:
            logger.error(f"Error getting station/grid for {airport_code}: {str(e)}")
            raise
    
    def _fetch_historical_observations(self,
                                    airport_code: str,
                                    start_date: datetime,
                                    end_date: datetime,
                                    use_cache: bool = True) -> pd.DataFrame:
        """Fetch historical weather observations for an airport."""
        cache_file = self.data_dir / f"hist_{airport_code}_{start_date.date()}_{end_date.date()}.parquet"
        
        # Check cache
        if use_cache and cache_file.exists():
            return pd.read_parquet(cache_file)
        
        try:
            station_id, _ = self._get_station_and_grid(airport_code)
            
            # Fetch observations
            url = f"{self.base_url}/stations/{station_id}/observations"
            params = {
                'start': start_date.isoformat(),
                'end': end_date.isoformat()
            }
            
            response = self.session.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Process observations
            observations = []
            for feature in data['features']:
                props = feature['properties']
                obs = {
                    'timestamp': pd.to_datetime(props['timestamp']),
                    'temperature': props['temperature']['value'],
                    'dewpoint': props['dewpoint']['value'],
                    'wind_speed': props['windSpeed']['value'],
                    'wind_direction': props['windDirection']['value'],
                    'humidity': props['relativeHumidity']['value'],
                    'precipitation_last_hour': props.get('precipitationLastHour', {}).get('value', 0),
                    'description': props.get('textDescription', '')
                }
                observations.append(obs)
            
            df = pd.DataFrame(observations)
            df.set_index('timestamp', inplace=True)
            
            # Convert units
            # Temperature: Celsius to Fahrenheit
            df['temperature'] = df['temperature'].apply(lambda x: x * 9/5 + 32 if pd.notnull(x) else x)
            df['dewpoint'] = df['dewpoint'].apply(lambda x: x * 9/5 + 32 if pd.notnull(x) else x)
            
            # Wind speed: m/s to mph
            df['wind_speed'] = df['wind_speed'].apply(lambda x: x * 2.237 if pd.notnull(x) else x)
            
            # Cache the data
            if use_cache:
                df.to_parquet(cache_file)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching historical data for {airport_code}: {str(e)}")
            raise
    
    def _fetch_forecast(self, airport_code: str) -> pd.DataFrame:
        """Fetch weather forecast for an airport."""
        try:
            _, grid = self._get_station_and_grid(airport_code)
            
            # Fetch hourly forecast
            url = f"{self.base_url}/gridpoints/{grid['grid_id']}/{grid['grid_x']},{grid['grid_y']}/forecast/hourly"
            response = self.session.get(url)
            response.raise_for_status()
            data = response.json()
            
            # Process forecast periods
            forecasts = []
            for period in data['properties']['periods']:
                forecast = {
                    'timestamp': pd.to_datetime(period['startTime']),
                    'temperature': period['temperature'],  # Already in Fahrenheit
                    'wind_speed': float(period['windSpeed'].split()[0]),  # Extract numeric value
                    'wind_direction': period['windDirection'],
                    'description': period['shortForecast'],
                    'precipitation_probability': period.get('probabilityOfPrecipitation', {}).get('value', 0)
                }
                forecasts.append(forecast)
            
            df = pd.DataFrame(forecasts)
            df.set_index('timestamp', inplace=True)
            
            return df
            
        except Exception as e:
            logger.error(f"Error fetching forecast for {airport_code}: {str(e)}")
            raise

if __name__ == "__main__":
    # Example usage
    collector = NOAACollector()
    
    try:
        # Fetch last 5 days of data
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)
        
        # Get data for all airports
        data = collector.fetch_data(start_date, end_date)
        
        # Print sample of data
        for airport in data:
            print(f"\n{airport} Historical Data:")
            print(data[airport]['historical'].head())
            
            if 'forecast' in data[airport]:
                print(f"\n{airport} Forecast Data:")
                print(data[airport]['forecast'].head())
        
    except Exception as e:
        print(f"Error: {str(e)}")