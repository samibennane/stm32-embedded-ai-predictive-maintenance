import serial
import numpy as np

PORT = "COM7"


def synchronise_UART(serial_port):
    """
    Synchronizes the UART communication by sending a byte and waiting for a response.

    Args:
        serial_port (serial.Serial): The serial port object to use for communication.

    Returns:
        None
    """
    while (1):
        serial_port.write(b"\xAB")
        ret = serial_port.read(1)
        if (ret == b"\xCD"):
            serial_port.read(1)
            break


def send_inputs_to_STM32(inputs, serial_port):
    """
    Sends a numpy array of inputs to the STM32 microcontroller via a serial port.

    Args:
        inputs (numpy.ndarray): The inputs to send to the STM32.
        serial_port (serial.Serial): The serial port to use for communication.

    Returns:
        None
    """
    inputs = inputs.astype(np.float32)
    buffer = b""
    for x in inputs:
        buffer += x.tobytes()
    serial_port.write(buffer)


def read_output_from_STM32(serial_port):
    """
    Reads 10 bytes from the given serial port and returns a list of float values obtained by dividing each byte by 255.

    Args:
    serial_port: A serial port object.

    Returns:
    A list of float values obtained by dividing each byte by 255.
    """
    output = serial_port.read(5)

    float_values = [int(out)/255 for out in output]
    return float_values


def evaluate_model_on_STM32(iterations, serial_port):

    accuracy = 0

    for i in range(iterations):

        print(f"----- Iteration {i+1} -----")

        send_inputs_to_STM32(X_test[i], serial_port)

        output = read_output_from_STM32(serial_port)

        if np.argmax(output) == Y_test[i]:
            accuracy += 1 / iterations

        print(f"Expected output: {Y_test[i]}")
        print(f"Received output: {output}")
        print(f"Accuracy: {accuracy:.2f}\n")

    return accuracy

if __name__ == '__main__':
    X_test = np.load("./machine_failure_xtest.npy")
    Y_test = np.load("./machine_failure_ytest.npy")

    with serial.Serial(PORT, 115200, timeout=1) as ser:
        print("Synchronising...")
        synchronise_UART(ser)
        print("Synchronised")

        print("Evaluating model on STM32...")
        error = evaluate_model_on_STM32(100, ser)