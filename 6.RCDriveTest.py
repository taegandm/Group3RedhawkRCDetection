import serial
import time
import serial.tools.list_ports

def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        print(p)
        return p[0]
    return None

arduino_port = find_arduino_port()

if arduino_port:
    try:
        ser = serial.Serial(arduino_port, 9600, timeout=1)
        print(f"Connected to Arduino on {arduino_port}")
        ser.flush()
        # You can now use 'ser' to communicate with your Arduino
    except Exception as e:
        print(f"Error connecting to Arduino: {e}")
else:
    print("Arduino not found")



def send_command(steering, velocity):
    command = f"{steering},{velocity}\n"
    ser.write(command.encode('utf-8'))
counter = 1
try:
    while True:
        # Example commands
        # Change these values to test different steering and velocity settings
        print("Test: ", counter)
        send_command(90, 100)  # Forward
        print("Forward")
        time.sleep(2)
        send_command(45, 100)  # Turn left
        print("Forward and left")
        time.sleep(2)
        send_command(135, 100) # Turn right
        print("Forward and right")
        time.sleep(2)
        send_command(90, 90)   # Stop
        print("Stop")
        time.sleep(2)
        counter+=1

except KeyboardInterrupt:
    # Stop the vehicle before exiting
    send_command(90, 90)
    print("Program terminated")
finally:
    ser.close()
