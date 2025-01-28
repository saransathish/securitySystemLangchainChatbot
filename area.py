import streamlit as st
import googlemaps
import pandas as pd
import numpy as np
from geopy.distance import geodesic
import requests
from datetime import datetime, timedelta
import os
from dotenv import load_dotenv
import folium
from streamlit_folium import folium_static
import re
from bs4 import BeautifulSoup
# Load environment variables
load_dotenv()

# Initialize Google Maps client
gmaps = googlemaps.Client(key=os.getenv('GOOGLE_MAPS_API_KEY'))

class DemographicAnalyzer:
    def __init__(self, api_key):
        self.gmaps = googlemaps.Client(key=api_key)

    def get_area_name(self, lat, lng):
        """Get area name from coordinates"""
        try:
            result = self.gmaps.reverse_geocode((lat, lng))
            for component in result[0]['address_components']:
                if 'postal_town' in component['types']:
                    return component['long_name']
                elif 'locality' in component['types']:
                    return component['long_name']
            return result[0]['formatted_address'].split(',')[0]
        except Exception as e:
            st.error(f"Error getting area name: {str(e)}")
            return None

    def get_demographic_data(self, area_name):
        """Get demographic data for an area"""
        try:
            # Search for population data
            population_query = f"{area_name} UK population statistics"
            population_result = self.gmaps.places(population_query)

            # Search for unemployment data
            unemployment_query = f"{area_name} UK unemployment rate"
            unemployment_result = self.gmaps.places(unemployment_query)

            # Search for student population
            student_query = f"{area_name} UK student population"
            student_result = self.gmaps.places(student_query)

            # Extract and process data
            # Note: In a real application, you would want to parse this data more carefully
            population = "Data available upon request"
            unemployment = "Data available upon request"
            student_pop = "Data available upon request"

            if population_result.get('results'):
                for result in population_result['results']:
                    if 'rating' in result:
                        population = f"~{result['user_ratings_total']:,} (estimated)"
                        break

            if unemployment_result.get('results'):
                for result in unemployment_result['results']:
                    if 'rating' in result:
                        unemployment = f"~{result['rating']}% (estimated)"
                        break

            if student_result.get('results'):
                for result in student_result['results']:
                    if 'user_ratings_total' in result:
                        student_pop = f"~{result['user_ratings_total']:,} (estimated)"
                        break

            return {
                'population': population,
                'unemployment_rate': unemployment,
                'student_population': student_pop
            }
        except Exception as e:
            st.error(f"Error fetching demographic data: {str(e)}")
            return None

class EnvironmentAnalyzer:
    def __init__(self, api_key):
        self.gmaps = googlemaps.Client(key=api_key)

    def analyze_environment(self, lat, lng, radius=2000):
        """Comprehensive environmental analysis"""
        try:
            # Get various place types
            tourist_attractions = self.get_places(lat, lng, 'tourist_attraction', radius)
            parks = self.get_places(lat, lng, 'park', radius)
            museums = self.get_places(lat, lng, 'museum', radius)
            restaurants = self.get_places(lat, lng, 'restaurant', radius)
            
            # Calculate tourism score
            tourism_score = self.calculate_tourism_score(
                tourist_attractions, parks, museums, restaurants
            )

            return {
                'tourist_attractions': tourist_attractions,
                'parks': parks,
                'museums': museums,
                'restaurants': restaurants,
                'tourism_score': tourism_score
            }
        except Exception as e:
            st.error(f"Error analyzing environment: {str(e)}")
            return None

    def get_places(self, lat, lng, place_type, radius):
        """Get places of specific type"""
        try:
            results = self.gmaps.places_nearby(
                location=(lat, lng),
                radius=radius,
                type=place_type
            )
            return results.get('results', [])
        except:
            return []

    def calculate_tourism_score(self, attractions, parks, museums, restaurants):
        """Calculate tourism score based on various factors"""
        score = 0
        
        # Weight different factors
        score += len(attractions) * 3  # Tourist attractions weighted heavily
        score += len(parks) * 1.5
        score += len(museums) * 2
        score += min(len(restaurants) * 0.5, 10)  # Cap restaurant score
        
        # Normalize to 0-100 scale
        normalized_score = min(score * 2, 100)
        
        if normalized_score >= 75:
            return "Very High"
        elif normalized_score >= 50:
            return "High"
        elif normalized_score >= 25:
            return "Medium"
        else:
            return "Low"
        
class CrimeAnalyzer:
    def get_available_date(self):
        """Get the most recent available date for crime data"""
        # Police UK API typically has 2-month delay
        current_date = datetime.now()
        # Start from 3 months ago to be safe
        check_date = current_date - timedelta(days=90)
        
        while check_date <= current_date:
            try:
                # Try with a test location (central London)
                test_url = "https://data.police.uk/api/crimes-street/all-crime"
                test_params = {
                    'lat': 51.5074,
                    'lng': -0.1278,
                    'date': check_date.strftime("%Y-%m")
                }
                response = requests.get(test_url, params=test_params)
                if response.status_code == 200:
                    return check_date.strftime("%Y-%m")
            except:
                pass
            check_date = check_date + timedelta(days=32)  # Move to next month
            check_date = check_date.replace(day=1)  # Reset to first of month
        
        return None

    def get_crimes_data(self, lat, lng, date=None):
        """Get crime data for a specific location"""
        try:
            if not date:
                date = self.get_available_date()
                if not date:
                    st.warning("Could not determine the latest available crime data date.")
                    return None
                
            url = "https://data.police.uk/api/crimes-street/all-crime"
            params = {
                'lat': lat,
                'lng': lng,
                'date': date
            }
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            st.error(f"Error fetching crime data: {str(e)}")
            return None

    def analyze_crimes(self, crimes_data):
        """Analyze crime data and return statistics"""
        if not crimes_data:
            return None

        df = pd.DataFrame(crimes_data)
        
        # Calculate statistics
        total_crimes = len(crimes_data)
        crimes_by_category = df['category'].value_counts()
        
        # Handle outcome status more carefully
        crimes_by_outcome = df['outcome_status'].apply(
            lambda x: x['category'] if isinstance(x, dict) and x is not None else 'Unknown'
        ).value_counts()

        return {
            'total_crimes': total_crimes,
            'crimes_by_category': crimes_by_category,
            'crimes_by_outcome': crimes_by_outcome,
            'crime_locations': df
        }

class LocationAnalyzer:
    def __init__(self, api_key):
        self.gmaps = googlemaps.Client(key=api_key)
        
    def get_location_coordinates(self, address):
        """Get coordinates for an address"""
        try:
            geocode_result = self.gmaps.geocode(address)
            if geocode_result:
                location = geocode_result[0]['geometry']['location']
                return location['lat'], location['lng']
        except Exception as e:
            st.error(f"Error geocoding address: {str(e)}")
        return None, None

    def get_nearby_places(self, location, place_type, radius, min_rating=0):
        """Search for nearby places with enhanced details"""
        lat, lng = location
        places = []
        
        try:
            results = self.gmaps.places_nearby(
                location=(lat, lng),
                radius=radius,
                type=place_type
            )
            
            if 'results' in results:
                for place in results['results']:
                    place_lat = place['geometry']['location']['lat']
                    place_lng = place['geometry']['location']['lng']
                    place_location = (place_lat, place_lng)
                    base_location = (lat, lng)
                    
                    distance = round(geodesic(base_location, place_location).miles, 2)
                    
                    place_details = self.gmaps.place(place['place_id'], 
                                                   fields=['name', 'rating', 'user_ratings_total',
                                                          'formatted_address', 'opening_hours',
                                                          'website'])
                    
                    place_info = {
                        'name': place['name'],
                        'address': place_details['result'].get('formatted_address', 'N/A'),
                        'distance': distance,
                        'rating': place_details['result'].get('rating', 'N/A'),
                        'total_ratings': place_details['result'].get('user_ratings_total', 0),
                        'website': place_details['result'].get('website', 'N/A'),
                        'lat': place_lat,
                        'lng': place_lng
                    }
                    
                    if place_info['rating'] != 'N/A' and place_info['rating'] >= min_rating:
                        places.append(place_info)
            
            return sorted(places, key=lambda x: x['distance'])
        
        except Exception as e:
            st.error(f"Error fetching places: {str(e)}")
            return []

    def analyze_motorway_junctions(self, lat, lng, radius=4828):
        """Analyze major motorway junctions"""
        motorways = ['M6', 'M1', 'M4', 'M5', 'M25', 'M62', 'M40', 'M8', 'M3', 
                    'M11', 'M20', 'M42', 'M74', 'M60', 'A1M']
        
        junctions = []
        for motorway in motorways:
            places = self.gmaps.places(
                f"{motorway} junction",
                location=(lat, lng),
                radius=radius
            )
            
            if 'results' in places:
                for place in places['results']:
                    place_lat = place['geometry']['location']['lat']
                    place_lng = place['geometry']['location']['lng']
                    distance = geodesic((lat, lng), (place_lat, place_lng)).miles
                    
                    if distance <= 3:
                        junctions.append({
                            'name': place['name'],
                            'motorway': motorway,
                            'distance': round(distance, 2)
                        })
        
        return junctions

def create_map(center_lat, center_lng, places_data, crimes_data=None):
    """Create a folium map with all locations and crimes marked"""
    m = folium.Map(location=[center_lat, center_lng], zoom_start=13)
    
    # Add center marker
    folium.Marker(
        [center_lat, center_lng],
        popup="Target Location",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    # Add places markers
    for place in places_data:
        folium.Marker(
            [place['lat'], place['lng']],
            popup=f"{place['name']}<br>Rating: {place['rating']}<br>Distance: {place['distance']} miles",
            icon=folium.Icon(color='blue')
        ).add_to(m)
    
    # Add crime markers
    if crimes_data:
        for crime in crimes_data:
            if 'location' in crime and 'latitude' in crime['location'] and 'longitude' in crime['location']:
                folium.CircleMarker(
                    location=[float(crime['location']['latitude']), 
                             float(crime['location']['longitude'])],
                    radius=8,
                    popup=f"Crime: {crime['category']}<br>Location: {crime['location']['street']['name']}",
                    color='red',
                    fill=True,
                    fill_color='red'
                ).add_to(m)
    
    return m

def main():
    st.title("Enhanced Area and Crime Analysis Tool")
    
    # User Information Section
    st.header("User Information")
    col1, col2 = st.columns(2)
    
    with col1:
        user_name = st.text_input("Full Name")
        business_name = st.text_input("Business Name")
    
    with col2:
        address = st.text_input("Business Address")
        radius = st.slider("Search Radius (miles)", 0.5, 3.0, 1.5) * 1609.34

    location_analyzer = LocationAnalyzer(os.getenv('GOOGLE_MAPS_API_KEY'))
    crime_analyzer = CrimeAnalyzer()
    demographic_analyzer = DemographicAnalyzer(os.getenv('GOOGLE_MAPS_API_KEY'))
    environment_analyzer = EnvironmentAnalyzer(os.getenv('GOOGLE_MAPS_API_KEY'))
    
    if st.button("Analyze Area") and address:
        lat, lng = location_analyzer.get_location_coordinates(address)
        
        if lat and lng:
            # Get area name for demographic data
            area_name = demographic_analyzer.get_area_name(lat, lng)
            demographic_data = demographic_analyzer.get_demographic_data(area_name)
            
            # Get environment data
            environment_data = environment_analyzer.analyze_environment(lat, lng)
            
            # Get crime data
            crimes_data = crime_analyzer.get_crimes_data(lat, lng)
            crime_analysis = crime_analyzer.analyze_crimes(crimes_data)
            
            # Create tabs for different categories
            tabs = st.tabs(["Crime Analysis", "Schools", "Retail & Shopping", 
                          "Transportation", "Demographics", "Environment", "Map View"])
            
            # Crime Analysis Tab
            with tabs[0]:
                st.subheader("Crime Analysis")
                if crime_analysis and crime_analysis['total_crimes'] > 0:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Total Crimes (Last Month)", 
                                crime_analysis['total_crimes'])
                        st.subheader("Crimes by Category")
                        st.bar_chart(crime_analysis['crimes_by_category'])
                    
                    with col2:
                        st.subheader("Crime Outcomes")
                        st.bar_chart(crime_analysis['crimes_by_outcome'])
                else:
                    st.info("No crime data found for this location in the selected time period.")
            
            # Schools Analysis
            with tabs[1]:
                st.subheader("Educational Institutions")
                schools = location_analyzer.get_nearby_places((lat, lng), 'school', radius)
                secondary_schools = [s for s in schools if 'secondary' in s['name'].lower() 
                                  or 'high school' in s['name'].lower()]
                
                st.metric("Secondary Schools within radius", len(secondary_schools))
                if schools:
                    df_schools = pd.DataFrame(schools)
                    st.dataframe(df_schools)
                    
                    school_types = {
                        'Secondary': len(secondary_schools),
                        'Primary': len([s for s in schools if 'primary' in s['name'].lower()]),
                        'College': len([s for s in schools if 'college' in s['name'].lower()]),
                        'Other': len(schools) - len(secondary_schools)
                    }
                    st.bar_chart(school_types)
            
            # Retail Analysis
            with tabs[2]:
                st.subheader("Retail & Shopping Analysis")
                retail_parks = location_analyzer.get_nearby_places((lat, lng), 'shopping_mall', radius)
                shopping_centers = location_analyzer.get_nearby_places((lat, lng), 'department_store', radius)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Retail Parks", len(retail_parks))
                with col2:
                    st.metric("Shopping Centers", len(shopping_centers))
                
                if retail_parks or shopping_centers:
                    combined_retail = pd.DataFrame(retail_parks + shopping_centers)
                    st.dataframe(combined_retail)
            
            # Transportation Analysis
            with tabs[3]:
                st.subheader("Transportation Analysis")
                transport_hubs = location_analyzer.get_nearby_places((lat, lng), 'transit_station', 4828)
                motorway_junctions = location_analyzer.analyze_motorway_junctions(lat, lng)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Transport Hubs", len(transport_hubs))
                with col2:
                    st.metric("Major Motorway Junctions", len(motorway_junctions))
                
                if transport_hubs:
                    st.subheader("Transport Hubs Details")
                    df_transport = pd.DataFrame(transport_hubs)
                    st.dataframe(df_transport)
                
                if motorway_junctions:
                    st.subheader("Motorway Junctions")
                    df_junctions = pd.DataFrame(motorway_junctions)
                    st.dataframe(df_junctions)
            
            # Demographics Analysis
            with tabs[4]:
                st.subheader("Demographics Analysis")
                if demographic_data:
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Population", demographic_data['population'])
                    with col2:
                        st.metric("Student Population", demographic_data['student_population'])
                    with col3:
                        st.metric("Unemployment Rate", demographic_data['unemployment_rate'])
                else:
                    st.info("Demographic data not available for this location.")

            # Environment Tab
            with tabs[5]:
                st.subheader("Environmental Analysis")
                if environment_data:
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Tourist Attractions", len(environment_data['tourist_attractions']))
                        st.metric("Parks", len(environment_data['parks']))
                    with col2:
                        st.metric("Museums", len(environment_data['museums']))
                        st.metric("Tourism Level", environment_data['tourism_score'])
                    
                    # Display top attractions
                    if environment_data['tourist_attractions']:
                        st.subheader("Notable Tourist Attractions")
                        attractions_df = pd.DataFrame([
                            {
                                'name': place['name'],
                                'rating': place.get('rating', 'N/A'),
                                'user_ratings': place.get('user_ratings_total', 'N/A')
                            }
                            for place in environment_data['tourist_attractions']
                        ])
                        st.dataframe(attractions_df)
                else:
                    st.info("Environmental data not available for this location.")
            
            # Map View
            with tabs[6]:
                st.subheader("Interactive Map View")
                all_places = schools + retail_parks + shopping_centers + transport_hubs
                m = create_map(lat, lng, all_places, crimes_data)
                folium_static(m)

if __name__ == "__main__":
    main()