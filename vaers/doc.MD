# vaers_data.CSV
| column name | type | description (it) | description (en) |
|------------|------|------------------|------------------|
| vaers_id | int64 | numero identificativo vaers | vaers identification number |
| recvdate | object | data di ricezione del report | date report was received |
| state | object | stato di residenza | state of residence |
| age_yrs | float64 | età in anni | age in years |
| cage_yr | float64 | età calcolata in anni | calculated age in years |
| cage_mo | float64 | età calcolata in mesi | calculated age in months |
| sex | object | sesso | sex |
| rpt_date | object | data di compilazione del modulo | date form completed |
| symptom_text | object | testo dei sintomi | symptom text |
| died | object | decesso | death indicator |
| datedied | object | data del decesso | date of death |
| l_threat | object | pericolo di vita | life threatening illness |
| er_visit | object | visita al pronto soccorso | emergency room visit |
| hospital | object | ospedalizzazione | hospitalized |
| hospdays | float64 | giorni di ospedalizzazione | days hospitalized |
| x_stay | object | prolungamento ospedalizzazione | prolonged hospitalization |
| disable | object | disabilità | disability |
| recovd | object | recupero | recovered |
| vax_date | object | data vaccinazione | vaccination date |
| onset_date | object | data insorgenza eventi | adverse event onset date |
| numdays | float64 | giorni tra vaccinazione e evento | days between vaccination and event |
| lab_data | object | dati di laboratorio | laboratory data |
| v_adminby | object | struttura di somministrazione | facility administering vaccine |
| v_fundby | object | tipo di finanziamento | type of funds |
| other_meds | object | altri farmaci | other medications |
| cur_ill | object | malattie attuali | current illnesses |
| history | object | storia clinica | medical history |
| prior_vax | object | vaccinazioni precedenti | prior vaccinations |
| splttype | object | numero produttore/progetto | manufacturer/project number |
| form_vers | int64 | versione del modulo | form version |
| todays_date | object | data odierna | today's date |
| birth_defect | object | difetti congeniti | birth defects |
| ofc_visit | object | visita ambulatoriale | office visit |
| er_ed_visit | object | visita al pronto soccorso | emergency room/department visit |
| allergies | object | allergie | allergies |

# vaers_symptomps.CSV
| column name | type | description (it) | description (en) |
|------------|------|------------------|------------------|
| vaers_id | int64 | numero identificativo vaers | vaers identification number |
| symptom1 | object | primo termine medra | first medra term |
| symptomversion1 | float64 | versione primo termine | first term version |
| symptom2 | object | secondo termine medra | second medra term |
| symptomversion2 | float64 | versione secondo termine | second term version |
| symptom3 | object | terzo termine medra | third medra term |
| symptomversion3 | float64 | versione terzo termine | third term version |
| symptom4 | object | quarto termine medra | fourth medra term |
| symptomversion4 | float64 | versione quarto termine | fourth term version |
| symptom5 | object | quinto termine medra | fifth medra term |
| symptomversion5 | float64 | versione quinto termine | fifth term version |

# vaers_vax.CSV
| column name | type | description (it) | description (en) |
|------------|------|------------------|------------------|
| vaers_id | int64 | numero identificativo vaers | vaers identification number |
| vax_type | object | tipo di vaccino | vaccine type |
| vax_manu | object | produttore vaccino | vaccine manufacturer |
| vax_lot | object | numero lotto | lot number |
| vax_dose_series | object | numero dose | dose number |
| vax_route | object | via di somministrazione | administration route |
| vax_site | object | sito di iniezione | injection site |
| vax_name | object | nome commerciale vaccino | vaccine brand name |
