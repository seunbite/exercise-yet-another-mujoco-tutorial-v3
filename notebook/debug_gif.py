#!/usr/bin/env python3

import os
import requests
from PIL import Image

def debug_gif(gif_url_or_path: str):
    """
    Debug GIF file to understand its properties and frame structure
    """
    print(f"🔍 Debugging GIF: {gif_url_or_path}")
    
    # Download if it's a URL
    if gif_url_or_path.startswith(('http://', 'https://')):
        print("📥 Downloading GIF...")
        response = requests.get(gif_url_or_path)
        gif_path = "debug_temp.gif"
        with open(gif_path, 'wb') as f:
            f.write(response.content)
        print(f"✅ Downloaded to {gif_path}")
    else:
        gif_path = gif_url_or_path
    
    try:
        with Image.open(gif_path) as gif:
            print(f"📊 GIF Properties:")
            print(f"   Size: {gif.size}")
            print(f"   Mode: {gif.mode}")
            print(f"   Format: {gif.format}")
            
            # Check if it's animated
            try:
                gif.seek(1)
                gif.seek(0)
                is_animated = True
                frame_count = getattr(gif, 'n_frames', 0)
                print(f"   🎬 Animated: Yes ({frame_count} frames)")
            except EOFError:
                is_animated = False
                print(f"   🖼️  Animated: No (static image)")
            
            # Get frame info
            if is_animated:
                print(f"   ⏱️  Frame info:")
                for i in range(min(5, frame_count)):  # Show first 5 frames
                    gif.seek(i)
                    duration = gif.info.get('duration', 0)
                    print(f"      Frame {i}: duration={duration}ms, mode={gif.mode}")
            
            # Try to extract a few frames manually
            print(f"\n🎯 Testing frame extraction:")
            os.makedirs("debug_frames", exist_ok=True)
            
            if is_animated:
                extracted = 0
                for i in range(min(10, frame_count)):  # Extract up to 10 frames
                    try:
                        gif.seek(i)
                        frame = gif.copy()
                        
                        # Convert based on mode
                        if frame.mode == 'P':
                            frame = frame.convert('RGBA')
                            background = Image.new('RGB', frame.size, (255, 255, 255))
                            if 'transparency' in frame.info:
                                background.paste(frame, mask=frame.split()[-1])
                            else:
                                background.paste(frame)
                            frame = background
                        else:
                            frame = frame.convert('RGB')
                        
                        frame.save(f"debug_frames/frame_{i:03d}.png")
                        extracted += 1
                    except Exception as e:
                        print(f"   ❌ Error extracting frame {i}: {e}")
                        break
                
                print(f"   ✅ Successfully extracted {extracted} frames to debug_frames/")
            else:
                # Static image
                frame = gif.convert('RGB')
                frame.save("debug_frames/frame_000.png")
                print(f"   ✅ Saved static frame to debug_frames/frame_000.png")
    
    except Exception as e:
        print(f"❌ Error analyzing GIF: {e}")
    
    # Clean up temp file
    if gif_url_or_path.startswith(('http://', 'https://')) and os.path.exists("debug_temp.gif"):
        os.remove("debug_temp.gif")

def test_gif_extraction():
    """
    Test the actual GIF extraction function
    """
    print("\n🧪 Testing GIF extraction function...")
    from retarget_data import extract_frames_from_gif
    
    gif_url = "https://i.makeagif.com/media/3-10-2023/oHTd3S.gif"
    
    # Download GIF
    response = requests.get(gif_url)
    with open("test.gif", 'wb') as f:
        f.write(response.content)
    
    # Test extraction
    result = extract_frames_from_gif("test.gif", "test_extracted_frames", fps=10)
    
    if result:
        frame_count = len([f for f in os.listdir("test_extracted_frames") if f.endswith('.png')])
        print(f"✅ Extraction successful: {frame_count} frames extracted")
    else:
        print("❌ Extraction failed")
    
    # Clean up
    if os.path.exists("test.gif"):
        os.remove("test.gif")

if __name__ == "__main__":
    # Debug the problematic GIF
    debug_gif("https://i.makeagif.com/media/3-10-2023/oHTd3S.gif")
    
    # Test our extraction function
    test_gif_extraction() 