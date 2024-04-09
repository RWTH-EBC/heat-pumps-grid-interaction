## Jahressimulationen

- Netze: 2 * 146 = ca. 292 Simulationen
	- Kerber 146 Gebäude
		- newbuildings
		- oldbuildings
	- IEEE? 55 Gebäude  --> Aus Kerber Netz entnehmen
		- oldbuildings
		- newbuildings
- Sanierungsquoten: 3
	- Unsaniert, Teilsaniert, Vollsaniert
- Wärmeerzeugung: 3 (bis 5)
	- Monovalent-WP
	- Monoenergetisch - WP+HS: Auslegung nach Regel TBiv = -6 °C
	- Hybrid - WP+Gas
		- Auslegung nach 65 % GEG (TBiv = 1 °C)
		- Auslegung Kostenoptimal? (TBiv = 14 °C) (wenig Mehrwert, da WP quasi nicht existent)
		- Auslegung auch nach Heizstab-Regel? (TBiv = 6 °C) (Einfache Abschätzung mit HS Ergebnissen möglich, allerdings nur ohne Batterie)
- Subsystem Strom: 1
	- Hausstrom (postprocessing)
	- Hausstrom + PV  (postprocessing)
	- Hausstrom + PV + Batterie
	- Hausstrom + E-Auto  (postprocessing)
	- Hausstrom + PV + E-Auto  (postprocessing)
	- Hausstrom + PV + Batterie + E-Auto (postprocessing)


## Monte-Carlo Simulationen

- Sanierungsquoten
- PV-Quoten
- PV+Batterie-Quoten
- E-Auto-Quoten
- Hybrid-Quoten
- WP-Quoten

## Netzsimulationen

- Relevante Ergebnisse aus den Monte-Carlo Simulationen
- Trafogrößen
	- 630
	- 1000
	- 2000?

## Sonderstudien

### Jahressimulationen
- Nachtabsenkung  --> Als default, mit Gaußverteilung
- Einfluss interne Gewinne  --> gering

### Netzsimulationen 
- Einfluss Gebäudeanordnung im Netz 
- Bedingt sich durch Quoten: Netzsimulationen mit
	- Hausstrom
	- Hausstrom + WP
	- Hausstrom + WP + E-Auto
	- Hausstrom + WP + E-Auto +  PV + Batterie


## Graphical Abstract ideas

- Central image: Kerber Netz (new / old buildings icon?!)
  - Old and new
- Step 1: BESMod Simulations --> X thousand simulations and X million possible system configurations
  - Different system configurations
    - Heat supply: fossil based, monovalent heat pump, bivalent hp with electric heater, hybrid heat pump
    - electric system: household, + pv, +pv+bat, +pv+bat+e-mob
    - Building envelope: standard, retrofit, advanced retrofit
- Step 2: Monte Carlo Analysis --> N thousand simulations to determine typical (median / mean maximal power) system configuration
  - With different quotas
- Step 3: Grid simulation of typical system configuration
  - With different trafo-sizes
