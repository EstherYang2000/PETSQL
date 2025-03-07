import argparse

def test_args():
    parser = argparse.ArgumentParser(description="Test argument parsing")
    parser.add_argument("--path_generate", type=str, help="Test path")
    parser.add_argument("--model", type=str, help="Test model")
    
    args = parser.parse_args()
    print(args)

if __name__ == '__main__':
    test_args()