def suggest_logo_name(category):
    # Define basic categories with responses
    suggestions = {
        "minimalist": "Sleek Designs",
        "technology": "CyberCode",
        "nature": "GreenLeaf",
        "fashion": "StyleLine",
        "sports": "ProPlay",
    }
    
    # Return suggestion or a default message
    return suggestions.get(category.lower(), "No suggestion available for this category.")

# Input from the user
category = input("Enter a category (e.g., minimalist, technology, nature, etc.): ")
print(suggest_logo_name(category))
