
import subprocess
import sys
import os
import shutil

print(f"Python: {sys.version}")
print(f"CWD: {os.getcwd()}")

# 1. Check if ffmpeg is in PATH
ffmpeg_path = shutil.which('ffmpeg')
if not ffmpeg_path:
    print("\n❌ ffmpeg executable NOT found in PATH!")
    print("Please install ffmpeg (e.g., 'conda install ffmpeg')")
    sys.exit(1)
print(f"\n✅ Found ffmpeg: {ffmpeg_path}")

# 2. Check version
print("\n=== ffmpeg Version ===")
try:
    res = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True)
    print(res.stdout.splitlines()[0])
except Exception as e:
    print(f"Error checking version: {e}")

# 3. Check codecs
print("\n=== Checking Codecs ===")
REQUIRED_CODECS = ['libx264', 'libopenh264', 'mpeg4', 'aac']
try:
    res = subprocess.run(['ffmpeg', '-codecs'], capture_output=True, text=True)
    available_codecs = res.stdout
    
    missing = []
    for c in REQUIRED_CODECS:
        if c in available_codecs:
            print(f"✅ {c}: Available")
        else:
            print(f"❌ {c}: MISSING")
            missing.append(c)
            
    if missing:
        print(f"\n⚠️ Missing codecs: {', '.join(missing)}")
        print("This may cause video generation using specific codecs to fail.")
    else:
        print("\n✅ All recommended codecs found.")
        
except Exception as e:
    print(f"Error checking codecs: {e}")

# 4. Test Generation (Dummy Video)
print("\n=== Testing Video Generation (libx264) ===")
output_file = "/tmp/test_ffmpeg_gen.mp4"
test_cmd = [
    'ffmpeg', '-y', 
    '-f', 'lavfi', '-i', 'testsrc=duration=1:size=640x480:rate=24',
    '-c:v', 'libx264', '-preset', 'ultrafast',
    '-pix_fmt', 'yuv420p',
    output_file
]
print(f"Command: {' '.join(test_cmd)}")

try:
    res = subprocess.run(test_cmd, capture_output=True, text=True)
    if res.returncode == 0 and os.path.exists(output_file):
        print("✅ Success! Generated video with libx264.")
        print(f"Size: {os.path.getsize(output_file)} bytes")
    else:
        print("❌ Failed to generate video with libx264.")
        print("Stderr output:")
        print(res.stderr)
        
        # Try fallback to mpeg4
        print("\n=== Testing Fallback (mpeg4) ===")
        test_cmd_fallback = [
            'ffmpeg', '-y', 
            '-f', 'lavfi', '-i', 'testsrc=duration=1:size=640x480:rate=24',
            '-c:v', 'mpeg4', '-q:v', '5',
            output_file
        ]
        res = subprocess.run(test_cmd_fallback, capture_output=True, text=True)
        if res.returncode == 0 and os.path.exists(output_file):
            print("✅ Success! Generated video with mpeg4.")
        else:
            print("❌ Failed fallback as well.")
            print(res.stderr)
            
except Exception as e:
    print(f"Execution error: {e}")
