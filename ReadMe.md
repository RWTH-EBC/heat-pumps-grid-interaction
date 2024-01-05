## Jahressimulationen

- Netze: 2 * 146 = ca. 292 Simulationen
	- Kerber 146 Gebäude
		- Neubau
		- Altbau
	- IEEE? 55 Gebäude  --> Aus Kerber Netz entnehmen
		- Altbau
		- Neubau
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
