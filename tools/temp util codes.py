


#script_execution_control

# This checks whether the current script is being run as the main program or if it is being imported as a module into another script.
# When a Python script is executed, Python sets the special variable __name__ to "__main__" for that script.
# If the script is imported as a module into another script, __name__ is set to the name of the module (e.g., the filename without the .py extension).
# Prevents Unintended Execution:
# If you import this script as a module in another script, the code inside the if __name__ == "__main__": block will not run. This prevents unintended execution of the main() function when the script is imported.

if __name__ == "__main__":
    main()



# HUGGING FACE LOGIN

login(token=hf_token)


# MODAL SECRETS
#Here's how you can pass huggingface-token to your Modal function:

import os
import modal

app = modal.App()

@app.function(secrets=[modal.Secret.from_name("huggingface-token")])
def f():
    print(os.environ["HF_TOKEN"])


#CONVERT PIL IMGS 

    # Convert PIL image to NumPy array
    numpy_array = np.array(image)
    print(numpy_array.shape)  # Should print (352, 640, 3) for height, width, channels
    # Convert PIL image to NumPy array
    numpy_array = np.array(pil_image)
    # Convert RGB to BGR for OpenCV
    opencv_image = cv2.cvtColor(numpy_array, cv2.COLOR_RGB2BGR)
    # Now you can use it with OpenCV functions
    cv2.imshow("OpenCV Image", opencv_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()