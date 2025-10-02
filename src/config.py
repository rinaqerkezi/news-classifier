
CATEGORY_MAPPING = {
    0: {"id": 1, "name": "World"},
    1: {"id": 2, "name": "Sports"}, 
    2: {"id": 3, "name": "Business"},
    3: {"id": 4, "name": "Sci/Tech"}
}

def get_category_name(category_id):
    """Convert category ID to name"""
    return CATEGORY_MAPPING.get(category_id, {"id": -1, "name": "Unknown"})