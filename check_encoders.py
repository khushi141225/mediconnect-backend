hospital_type_map = {"Government": 0, "Private": 1}
yes_no_map = {"No": 0, "Yes": 1}

hospital_type_encoded = hospital_type_map[data.hospital_type]
specialization_encoded = yes_no_map[data.specialization_available]
emergency_encoded = yes_no_map[data.emergency_services]