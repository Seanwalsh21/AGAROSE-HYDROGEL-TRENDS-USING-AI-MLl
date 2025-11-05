from ilastik_data_loader import IlastikDataMapper
import inspect

def debug_data_loader():
    """complete debugging of the data loader"""
    
    # Initialize mapper
    mapper = IlastikDataMapper()
    
    # Check mapper attributes
    print(f"1. MAPPER OBJECT INSPECTION:")
    print(f"   Type: {type(mapper)}")
    print(f"   Attributes:")
    
    for attr in dir(mapper):
        if not attr.startswith('_'):
            try:
                value = getattr(mapper, attr)
                if not callable(value):
                    print(f"   - {attr}: {value}")
            except:
                pass
    
    # Check mapper methods
    print(f"2. MAPPER METHODS:")
    for attr in dir(mapper):
        if not attr.startswith('_') and callable(getattr(mapper, attr)):
            print(f"   - {attr}()")
    
    # Try to load data pairs
    print(f"\n3. LOADING DATA PAIRS FOR AFM:")
    try:
        pairs = mapper.load_data_pairs('AFM')
        print(f"Pairs loaded: {len(pairs)}")
        
        if pairs:
            print(f"\n   First 3 pairs:")
            for i, (img, lbl) in enumerate(pairs[:3]):
                print(f"   [{i}]")
                print(f"      Image: {img}")
                print(f"      Label: {lbl}")
        else:
            print("NO PAIRS FOUND!")
            
    except Exception as e:
        print(f"ERROR loading pairs: {e}")
        import traceback
        traceback.print_exc()
    
    # Try to inspect the load_data_pairs method
    print(f"4. INSPECTING load_data_pairs METHOD:")
    try:
        sig = inspect.signature(mapper.load_data_pairs)
        print(f"   Signature: {sig}")
        
        # Get the source code if possible
        try:
            source = inspect.getsource(mapper.load_data_pairs)
            print(f"\n   Source code:")
            print("   " + "\n   ".join(source.split('\n')[:30]))  # First 30 lines
        except:
            print("(Source code not available)")
            
    except Exception as e:
        print(f"error inspecting method: {e}")
    
    # Try other common method names
    print(f"5. TESTING COMMON METHOD PATTERNS:")
    
    test_methods = [
        ('get_image_files', 'AFM'),
        ('get_label_files', 'AFM'),
        ('get_data_pairs', 'AFM'),
        ('load_images', 'AFM'),
        ('load_labels', 'AFM'),
    ]
    
    for method_name, arg in test_methods:
        if hasattr(mapper, method_name):
            try:
                result = getattr(mapper, method_name)(arg)
                if isinstance(result, list):
                    print(f"{method_name}('{arg}'): {len(result)} items")
                    if result and len(result) > 0:
                        print(f"      First item: {result[0]}")
                else:
                    print(f"{method_name}('{arg}'): {result}")
            except Exception as e:
                print(f"{method_name}('{arg}'): {e}")
        else:
            print(f"{method_name}: method not found")

if __name__ == "__main__":
    debug_data_loader()