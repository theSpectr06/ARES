# ARES — Advanced Route Exploration System

[![NASA Space Apps](https://img.shields.io/badge/NASA-Space%20Apps%202026-0B3D91?logo=nasa&logoColor=white)](https://www.spaceappschallenge.org/)
[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![Rasterio](https://img.shields.io/badge/Rasterio-geospatial-2C7A7B)](https://rasterio.readthedocs.io/)
[![NumPy](https://img.shields.io/badge/NumPy-computing-013243?logo=numpy&logoColor=white)](https://numpy.org/)
[![SciPy](https://img.shields.io/badge/SciPy-scientific%20computing-8CAAE6?logo=scipy&logoColor=white)](https://scipy.org/)
[![Tests](https://img.shields.io/badge/tests-57%20passing-brightgreen?logo=pytest&logoColor=white)](https://docs.pytest.org/)

**Terrain-aware route planning for future human exploration of Mars.**

ARES is a Mars surface exploration planning prototype developed for the [NASA Space Apps Challenge 2026](https://www.spaceappschallenge.org/).

It uses high-resolution Martian terrain data to analyse traversability and calculate routes between selected locations. The goal is to help mission planners understand terrain constraints while planning exploration objectives.

## Current capabilities

- HiRISE DEM, orthoimage and confidence raster alignment.
- Terrain slope and detrended roughness analysis.
- Combined terrain traversal-cost generation.
- Eight-directional A* pathfinding with terrain-weighted costs.
- GeoPackage export of generated routes.

## Data source

**Mars Sample Return Terrain Relative Navigation HiRISE DTM and Orthoimage Mosaics**

- Provider: USGS / NASA Jet Propulsion Laboratory
- Instrument: HiRISE aboard Mars Reconnaissance Orbiter
- DOI: [10.5066/P13CPYYU](https://doi.org/10.5066/P13CPYYU)

## Technology

Python · Rasterio · NumPy · SciPy · Shapely · Fiona · pytest

## Project status

The geospatial processing pipeline and terrain-aware pathfinding are implemented and tested. The interactive mission-planning interface and additional scientific context are under development.

## Scope and limitations

ARES is an experimental planning prototype, not a certified astronaut navigation or hazard-detection system. Its terrain-cost model uses derived slope and roughness metrics and does not detect every surface hazard.

## Acknowledgements

Developed using openly available planetary data and software. AI tools assisted parts of the development process.
