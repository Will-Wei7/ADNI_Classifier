from transformers import ViTForImageClassification

model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224-in21k")

print("Modules containing 'attention':")
for name, _ in model.named_modules():
    if 'attention' in name:
        print(f"  - {name}")

print("\nModules related to query, key, value:")
for name, _ in model.named_modules():
    if any(term in name for term in ['query', 'key', 'value', 'qkv']):
        print(f"  - {name}")

print("\nModules related to projections:")
for name, _ in model.named_modules():
    if any(term in name for term in ['proj', 'projection']):
        print(f"  - {name}") 