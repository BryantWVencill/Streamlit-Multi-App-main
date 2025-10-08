import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz

# --- Page Configuration ---
st.set_page_config(
    page_title="Convolutional Neural Networks Explained",
    page_icon="🧠",
    layout="wide",
)

# --- Helper Functions ---

def plot_relu():
    """Generates and returns a matplotlib figure of the ReLU function."""
    x = np.linspace(-10, 10, 100)
    y = np.maximum(0, x)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(x, y)
    ax.set_title("ReLU Activation Function")
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    ax.grid(True)
    return fig

def create_convolution_matrix(input_shape, kernel_shape, kernel):
    """Constructs the matrix for a 2D cross-correlation operation."""
    i_h, i_w = input_shape
    k_h, k_w = kernel_shape
    o_h, o_w = i_h - k_h + 1, i_w - k_w + 1
    
    conv_matrix = np.zeros((o_h * o_w, i_h * i_w))
    
    for oh_idx in range(o_h):
        for ow_idx in range(o_w):
            output_pixel_idx = oh_idx * o_w + ow_idx
            for kh_idx in range(k_h):
                for kw_idx in range(k_w):
                    ih_idx, iw_idx = oh_idx + kh_idx, ow_idx + kw_idx
                    input_pixel_idx = ih_idx * i_w + iw_idx
                    conv_matrix[output_pixel_idx, input_pixel_idx] = kernel[kh_idx, kw_idx]
    return conv_matrix

def show_rgb_example():
    """Creates an interactive visualization of an RGB image and its channels."""
    st.subheader("Interactive RGB Image Example")
    st.markdown("An RGB image is a 3D array of pixels: `Height x Width x 3 Channels`.")
    
    # Use session state to keep the image consistent across reruns
    if 'rgb_image' not in st.session_state:
        img = np.zeros((4, 4, 3), dtype=np.uint8)
        img[0:2, 0:2] = [255, 0, 0]    # Red square
        img[0:2, 2:4] = [0, 255, 0]    # Green square
        img[2:4, 0:2] = [0, 0, 255]    # Blue square
        img[2:4, 2:4] = [255, 255, 0]  # Yellow square
        st.session_state.rgb_image = img

    image = st.session_state.rgb_image
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**Full RGB Image**")
        st.image(image, caption=f"Shape: {image.shape}", use_column_width='always')
    with col2:
        st.markdown("**Red Channel**")
        st.image(image[:, :, 0], caption=f"Shape: {image[:, :, 0].shape}", use_column_width='always')
    with col3:
        st.markdown("**Green Channel**")
        st.image(image[:, :, 1], caption=f"Shape: {image[:, :, 1].shape}", use_column_width='always')
    with col4:
        st.markdown("**Blue Channel**")
        st.image(image[:, :, 2], caption=f"Shape: {image[:, :, 2].shape}", use_column_width='always')
    st.info("A CNN processes these three channels together as a single 3D volume of data.")

def plot_softmax_output():
    """Generates a bar chart representing softmax output."""
    classes = ['Cat', 'Dog', 'Bird']
    probabilities = np.array([0.85, 0.1, 0.05])
    
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.bar(classes, probabilities, color=['#ff9999','#66b3ff','#99ff99'])
    ax.set_ylabel('Probability')
    ax.set_title('Example Softmax Output')
    ax.set_ylim(0, 1)
    for i, v in enumerate(probabilities):
        ax.text(i, v + 0.02, f"{v*100:.0f}%", ha='center', fontweight='bold')
    return fig


# --- Main App ---

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Go to",
    ("Introduction", "Part 1: The Math of Convolution", "Part 2: Understanding CNNs", "Ethical Considerations"),
)
st.sidebar.info("An interactive guide to the fundamentals of Convolutional Neural Networks.")

# --- Page 1: Introduction ---
if page == "Introduction":
    st.title("Assignment: Convolutional Neural Networks (CNNs)")
    st.header("Project Overview")
    st.markdown(
        """
        This app is an interactive guide to CNNs for the assignment, covering the core concepts from the underlying math to the final architecture and ethical implications.
        
        Use the sidebar to navigate through the different sections.
        """
    )

# --- Page 2: The Math of Convolution ---
elif page == "Part 1: The Math of Convolution":
    st.title("Part 1: The Math of Convolution")
    st.markdown("CNNs use a **convolution operation**, where a **kernel** (or filter) slides over an input matrix to create a **feature map** that highlights patterns.")
    st.header("Interactive Convolution Demo")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("1. Input Matrix")
        input_matrix_str = st.text_area("Enter the input matrix:", "1,1,1,0,0\n0,1,1,1,0\n0,0,1,1,1\n0,0,1,1,0\n0,1,1,0,0", height=150)
        try:
            input_matrix = np.array([list(map(int, r.split(','))) for r in input_matrix_str.strip().split('\n')])
            st.table(input_matrix)
        except:
            st.error("Invalid input matrix format.")
            input_matrix = None
    with col2:
        st.subheader("2. Kernel (Filter)")
        kernel_str = st.text_area("Enter the kernel matrix:", "1,0,-1\n1,0,-1\n1,0,-1", height=100)
        try:
            kernel = np.array([list(map(int, r.split(','))) for r in kernel_str.strip().split('\n')])
            st.table(kernel)
        except:
            st.error("Invalid kernel format.")
            kernel = None
    
    if input_matrix is not None and kernel is not None and st.button("Calculate Convolution"):
        i_h, i_w = input_matrix.shape
        k_h, k_w = kernel.shape
        o_h, o_w = i_h - k_h + 1, i_w - k_w + 1
        output_matrix = np.zeros((o_h, o_w))
        
        st.header("Results")
        st.subheader("Step-by-Step Calculation")
        
        expander = st.expander("Show step-by-step calculation")

        for y in range(o_h):
            for x in range(o_w):
                roi = input_matrix[y:y+k_h, x:x+k_w]
                output_matrix[y, x] = np.sum(roi * kernel)
                with expander:
                    st.markdown(f"**Step ({y+1}, {x+1}):**")
                    st.text(f"Region of Interest:\n{roi}")
                    st.text(f"Kernel:\n{kernel}")
                    st.text(f"Calculation: np.sum({roi.flatten()} * {kernel.flatten()}) = {output_matrix[y, x]}")
                    st.markdown("---")

        st.subheader("3. Resulting Feature Map")
        st.table(output_matrix.astype(int))
        
        st.subheader("4. The Convolution Matrix")
        st.markdown("The entire operation can be represented as `Output = Matrix * Input`.")
        try:
            conv_matrix = create_convolution_matrix(input_matrix.shape, kernel.shape, kernel)
            output_from_matrix = (conv_matrix @ input_matrix.flatten()).reshape((o_h, o_w))
            st.dataframe(conv_matrix.astype(int))
            st.markdown("**Verification:**")
            if np.allclose(output_matrix, output_from_matrix):
                st.success("Verification successful: The outputs from both methods match!")
            else:
                st.error("Verification failed.")
        except Exception as e:
            st.error(f"Could not generate the convolution matrix: {e}")

# --- Page 3: Understanding CNNs ---
elif page == "Part 2: Understanding CNNs":
    st.title("Part 2: Understanding CNNs")
    st.markdown("CNNs are deep learning models for grid-like data (e.g., images). They learn a hierarchy of features, from simple edges to complex objects.")
    st.image("https://miro.medium.com/v2/resize:fit:1400/1*uAeANQIOuKSqnuFJ-vWwWg.png", caption="A typical CNN architecture.", use_column_width=True)

    tab1, tab2 = st.tabs(["Image Dimensions", "Steps in a CNN"])
    with tab1:
        show_rgb_example()
        st.subheader("Grayscale Images")
        st.markdown("Grayscale images are simpler, with only 2 dimensions (`Height x Width`) and 1 channel.")

    with tab2:
        st.subheader("Deep Dive: The Steps in a CNN")
        st.markdown("#### 1. Convolutional Layers")
        st.markdown("Filters (kernels) slide over the input image, performing element-wise multiplication and summation to create a **feature map**. Each map highlights a specific feature.")
        st.image("https://i.imgur.com/1Abh2xI.gif", caption="Animation of a filter creating a feature map.", use_column_width=True)

        st.markdown("#### 2. Activation Function (ReLU)")
        st.markdown("The **ReLU** function introduces non-linearity by changing all negative values to zero (`f(x) = max(0, x)`), allowing the network to learn more complex patterns.")
        st.pyplot(plot_relu())

        st.markdown("#### 3. Pooling Layers")
        st.markdown("**Pooling layers** reduce the size of feature maps to save computation and make the model more robust. **Max Pooling** is most common, taking the max value from each window.")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Input (4x4)**")
            st.table(np.array([[1, 4, 2, 8], [3, 7, 0, 5], [6, 2, 9, 1], [5, 3, 4, 6]]))
        with col2:
            st.markdown("**Output after 2x2 Max Pooling (2x2)**")
            st.table(np.array([[7, 8], [6, 9]]))

        st.markdown("#### 4. Fully Connected Layers & Output")
        st.markdown("After feature extraction, the final maps are **flattened** into a vector and fed into **Fully Connected Layers** for classification. The **Output Layer** uses a function like **Softmax** to convert the final scores into class probabilities.")
        st.pyplot(plot_softmax_output())

# --- Page 4: Ethical Considerations ---
elif page == "Ethical Considerations in CNNs":
    st.title("Ethical Considerations in CNNs")
    st.markdown("Developing and deploying CNNs comes with significant ethical responsibilities.")
    
    st.subheader("1. Biases in Image Datasets")
    st.warning("""
    **Concern:** The performance and fairness of a CNN are highly dependent on the data it was trained on. If a training dataset is not diverse and representative of the real world, the model will inherit and amplify existing societal biases.
    
    **Example:** A facial recognition model trained primarily on images of one demographic may perform poorly and make unfair judgments when applied to individuals from underrepresented groups.
    """)

    st.subheader("2. Security and Confidentiality of Data")
    st.warning("""
    **Concern:** CNNs often require vast amounts of data for training, which can include sensitive personal information such as medical scans or personal photos for facial recognition.
    
    **Example:** A breach of a healthcare database used for training a medical imaging CNN could expose the private health information of thousands of patients.
    """)
    
    st.subheader("3. Need for Interpretability and Transparency")
    st.warning("""
    **Concern:** CNNs are often considered "black boxes" because their decision-making processes are not easily understood by humans. It can be difficult to determine *why* a model made a particular prediction.
    
    **Example:** If a CNN denies a loan application, the applicant has a right to know the reasoning behind the decision. Without transparency, it's impossible to challenge or correct erroneous conclusions.
    """)
    
    st.subheader("4. Responsible Deployment in Critical Applications")
    st.warning("""
    **Concern:** The deployment of CNNs in high-stakes applications like autonomous vehicle navigation, medical diagnosis, and public surveillance carries significant risks.
    
    **Example:** An autonomous car's object detection system failing to recognize a pedestrian in unusual lighting conditions could have fatal consequences.
    """)
    
    st.subheader("5. Potential Harmful Consequences of Errors")
    st.warning("""
    **Concern:** Even highly accurate models make mistakes. The consequences of these inaccuracies can range from minor inconveniences to severe harm.
    
    **Example:** A diagnostic AI incorrectly identifying cancer in a medical scan can lead to immense patient distress and unnecessary procedures. Conversely, failing to detect a disease could delay critical treatment.
    """)

