
def fix_unidecode(result):
    for key, value in result.items():
        if isinstance(value, list):
            # Apply the transformation to each element in the list if it is a string
            result[key] = [
                item.encode('utf-8').decode('unicode_escape') if isinstance(item, str) else item
                for item in value
            ]
        elif isinstance(value, str):
            result[key] = value.encode('utf-8').decode('unicode_escape')
        else:
            # If it's another type, keep the original value
            result[key] = value
    return result