def create_mmsi_dict_from_file(file_path):
    mmsi_type_dict = {}
    
    try:
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    mmsi_part, type_part = line.split(',', 1)

                    mmsi_key = mmsi_part.split(':', 1)[1].strip()
                    ship_type_value = type_part.split(':', 1)[1].strip()

                    mmsi_type_dict[mmsi_key] = ship_type_value
                    
                except (ValueError, IndexError):
                    print(f"Skipping malformed line: '{line}'")

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    return mmsi_type_dict


    