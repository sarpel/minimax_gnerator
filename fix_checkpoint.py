# Read the file
with open("wakegen/generation/checkpoint.py", encoding="utf-8") as f:
    content = f.read()

# Fix escaped triple quotes
content = content.replace('\\"\\"\\"', '"""')

# Write back
with open("wakegen/generation/checkpoint.py", "w", encoding="utf-8") as f:
    f.write(content)

print("Fixed escaped quotes in checkpoint.py")
