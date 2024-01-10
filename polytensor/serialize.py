import json


def tojson(poly):
    """Extracts poly.tensor and finds all nonzero elements and their indices.
    Returns a json string with the nonzero elements and their indices.
    """

    # Get all nonzero elements and their indices
    nonzero = poly.tensor.nonzero()
    indices = nonzero[0].tolist()
    values = poly.tensor[nonzero].tolist()

    # Create a dictionary with the indices and values
    data = {}
    for i in range(len(indices)):
        data[indices[i]] = values[i]

    # Return the json string
    return json.dumps(data)
