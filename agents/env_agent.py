import asyncio
import platform
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import List, Dict
import numpy as np
from sklearn.linear_model import LinearRegression
from pulp import LpProblem, LpMinimize, LpVariable, lpSum, PULP_CBC_CMD
import random

# Mock data for Greenview (July 26, 2025, 18:48 IST)
@dataclass
class AirQuality:
    district: str
    pm25: float  # µg/m³
    no2: float   # µg/m³
    co: float    # ppm
    notes: str

@dataclass
class EnergySource:
    name: str
    output: float  # MW
    max_capacity: float  # MW
    cost: float  # $/MWh
    pollution: float  # kg CO₂/MWh

# WHO thresholds
THRESHOLDS = {"pm25": 25, "no2": 40, "co": 7}

# Mock air quality data (initial)
air_quality_data = [
    AirQuality("Downtown", 35, 50, 8, "Industrial area, traffic"),
    AirQuality("Northside", 20, 30, 5, "Residential, green spaces"),
    AirQuality("Eastport", 40, 45, 9, "Port, heavy vehicles"),
    AirQuality("Westview", 15, 25, 4, "Suburban, low traffic"),
    AirQuality("Southend", 30, 55, 7, "Factories, moderate traffic")
]

# Mock energy grid data
energy_sources = [
    EnergySource("Solar", 50, 80, 40, 0),
    EnergySource("Wind", 30, 50, 50, 0),
    EnergySource("Coal", 100, 200, 70, 800),
    EnergySource("Natural Gas", 80, 150, 60, 400)
]

# Mock historical PM2.5 data (last 24 hours, hourly)
historical_data = {
    district.district: [district.pm25 + random.uniform(-5, 5) for _ in range(24)]
    for district in air_quality_data
}

# Current and forecast demand
current_demand = 220  # MW
forecast_demand = 250  # MW

class EnvironmentalAgent:
    def __init__(self, air_quality: List[AirQuality], energy_sources: List[EnergySource]):
        self.air_quality = air_quality
        self.energy_sources = energy_sources
        self.hotspots = []
        self.models = {}
        self.train_ml_models()

    def train_ml_models(self):
        """Train linear regression models for PM2.5 prediction per district."""
        hours = np.array(range(24)).reshape(-1, 1)
        for district in historical_data:
            pm25 = np.array(historical_data[district]).reshape(-1, 1)
            model = LinearRegression()
            model.fit(hours, pm25)
            self.models[district] = model

    def predict_pm25(self, district: str, hours_ahead: int = 1) -> float:
        """Predict PM2.5 levels for a district."""
        return float(self.models[district].predict([[24 + hours_ahead]])[0])

    def identify_hotspots(self) -> List[str]:
        """Identify districts with pollutant levels exceeding WHO thresholds."""
        self.hotspots = []
        for data in self.air_quality:
            if (data.pm25 > THRESHOLDS["pm25"] or 
                data.no2 > THRESHOLDS["no2"] or 
                data.co > THRESHOLDS["co"]):
                self.hotspots.append(data.district)
        print(f"Hotspots identified: {self.hotspots}")
        return self.hotspots

    def optimize_energy_grid(self, demand: float) -> Dict[str, float]:
        """Optimize energy allocation using linear programming."""
        prob = LpProblem("Energy_Optimization", LpMinimize)
        allocations = {source.name: LpVariable(source.name, 0, source.max_capacity) for source in self.energy_sources}
        
        # Objective: Minimize cost + weighted pollution
        prob += lpSum([source.cost * allocations[source.name] + 
                       source.pollution * allocations[source.name] * 0.1 
                       for source in self.energy_sources])
        
        # Constraint: Meet demand
        prob += lpSum([allocations[source.name] for source in self.energy_sources]) == demand
        
        # Solve
        prob.solve(PULP_CBC_CMD(msg=False))
        allocation = {name: var.varValue for name, var in allocations.items()}
        
        print(f"Energy allocation: {allocation}")
        return allocation

    def update_air_quality(self):
        """Simulate real-time air quality updates with mock data."""
        for data in self.air_quality:
            data.pm25 += random.uniform(-2, 2)  # Simulate fluctuations
            data.no2 += random.uniform(-3, 3)
            data.co += random.uniform(-0.5, 0.5)
            # Ensure non-negative values
            data.pm25 = max(0, data.pm25)
            data.no2 = max(0, data.no2)
            data.co = max(0, data.co)

    def plot_hotspots_and_trends(self):
        """Generate bar chart for PM2.5 and line plot for predicted trends."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Bar chart for current PM2.5
        districts = [data.district for data in self.air_quality]
        pm25_levels = [data.pm25 for data in self.air_quality]
        colors = ['red' if district in self.hotspots else 'green' for district in districts]
        ax1.bar(districts, pm25_levels, color=colors)
        ax1.axhline(y=THRESHOLDS["pm25"], color='black', linestyle='--', label='WHO PM2.5 Threshold')
        ax1.set_xlabel("Districts")
        ax1.set_ylabel("PM2.5 (µg/m³)")
        ax1.set_title("Current Pollution Hotspots")
        ax1.legend()
        
        # Line plot for PM2.5 predictions (next 6 hours)
        future_hours = np.array(range(1, 7)).reshape(-1, 1)
        for district in districts:
            predictions = [self.predict_pm25(district, h) for h in range(1, 7)]
            ax2.plot(future_hours, predictions, label=district)
        ax2.set_xlabel("Hours Ahead")
        ax2.set_ylabel("Predicted PM2.5 (µg/m³)")
        ax2.set_title("PM2.5 Trend Forecast")
        ax2.legend()
        
        plt.tight_layout()
        plt.show()

async def main():
    agent = EnvironmentalAgent(air_quality_data, energy_sources)
    
    # Simulate real-time monitoring
    for _ in range(3):  # Limit to 3 cycles for demo
        print("\n--- Environmental Agent Update (July 26, 2025, 18:48 IST) ---")
        agent.update_air_quality()
        agent.identify_hotspots()
        agent.optimize_energy_grid(forecast_demand)
        agent.plot_hotspots_and_trends()
        print("Waiting for next update cycle...")
        await asyncio.sleep(60)  # Simulate 1-minute updates

if platform.system() == "Emscripten":
    asyncio.ensure_future(main())
else:
    if __name__ == "__main__":
        asyncio.run(main())