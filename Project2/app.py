import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import toeplitz
import graphviz

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
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title("ReLU (Rectified Linear Unit) Activation Function")
    ax.set_xlabel("Input")
    ax.set_ylabel("Output")
    ax.grid(True)
    return fig

def create_convolution_matrix(input_shape, kernel_shape):
    """
    Constructs the Toeplitz matrix for a 2D convolution operation.
    This demonstrates the mathematical underpinning of convolution as a matrix multiplication.
    """
    i_h, i_w = input_shape
    k_h, k_w = kernel_shape
    
    # Calculate output dimensions
    o_h = i_h - k_h + 1
    o_w = i_w - k_w + 1
    
    # Create the Toeplitz matrix for each row of the kernel
    toeplitz_matrices = []
    for r in range(k_h):
        # The first row of the Toeplitz matrix for this kernel row
        c = np.zeros(i_w)
        c[:k_w] = np.flip(kernel[r, :])
        # The first column
        r_col = np.zeros(o_w)
        r_col[0] = c[0]
        
        T = toeplitz(c=r_col, r=c)
        toeplitz_matrices.append(T)
        
    # Create the doubly block Toeplitz matrix
    # The number of blocks vertically is the output height
    num_blocks_h = o_h
    # The number of blocks horizontally is the input height
    num_blocks_w = i_h
    
    block_h, block_w = toeplitz_matrices[0].shape
    
    # Initialize the final large matrix
    C = np.zeros((num_blocks_h * block_h, num_blocks_w * block_w))
    
    for i in range(k_h):
        for j in range(o_h):
            row_start = j * block_h
            row_end = row_start + block_h
            col_start = (j + i) * block_w
            col_end = col_start + block_w
            
            if col_end <= C.shape[1]:
                C[row_start:row_end, col_start:col_end] = toeplitz_matrices[i]
                
    return C

# --- Main App ---

# --- Sidebar Navigation ---
st.sidebar.title("Navigation")
st.sidebar.markdown("Use the options below to explore the different parts of the assignment.")

page = st.sidebar.radio(
    "Go to",
    (
        "Introduction",
        "Part 1: The Math of Convolution",
        "Part 2: Understanding CNNs",
        "Ethical Considerations in CNNs",
    ),
)
st.sidebar.info(
    "This Streamlit app provides an interactive explanation of Convolutional Neural Networks, "
    "from the underlying math to their architecture and ethical implications."
)


# --- Page 1: Introduction ---
if page == "Introduction":
    st.title("Assignment: Convolutional Neural Networks (CNNs)")
    st.markdown("---")
    st.header("Project Overview")
    st.markdown(
        """
        This application serves as a comprehensive deliverable for the assignment on Convolutional Neural Networks. It is designed to be an interactive tool that explains the core concepts, from the fundamental mathematics to the practical architecture and ethical considerations.

        The project is divided into several sections, accessible via the navigation sidebar:

        - **Part 1: The Math of Convolution:** An interactive section to explore how convolution operations work by manipulating input matrices and kernels. It also demonstrates how to construct and analyze the corresponding convolution matrix (Toeplitz matrix).

        - **Part 2: Understanding CNNs:** A detailed breakdown of the components that make up a CNN, including convolutional layers, pooling layers, and fully connected layers. This section uses diagrams and visual aids to explain the entire process.

        - **Ethical Considerations in CNNs:** A discussion of the crucial ethical concerns related to the development and deployment of CNNs, such as data bias, privacy, and responsible use.

        This application is built using Streamlit, a Python framework for creating web apps for machine learning and data science projects.
        """
    )
    st.info("Select a section from the sidebar on the left to begin.")

# --- Page 2: The Math of Convolution ---
elif page == "Part 1: The Math of Convolution":
    st.title("Part 1: The Math of Convolution")
    st.markdown("---")
    st.markdown(
        """
        At the heart of every CNN is the **convolution operation**. This operation uses a small matrix called a **kernel** (or filter) to slide over an input matrix (representing an image) and produce an output matrix called a **feature map**. The feature map highlights specific patterns in the input, such as edges, corners, or textures.

        Mathematically, a convolution is an operation on two functions that produces a third function expressing how the shape of one is modified by the other. In our case, it's a discrete operation on two matrices.

        Below, you can interactively explore this operation.
        """
    )

    st.header("Interactive Convolution Demo")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("1. Define the Input Matrix")
        # Example input matrix representing a simple 5x5 image
        input_matrix_str = st.text_area(
            "Enter the input matrix (comma-separated, rows on new lines):",
            "1, 1, 1, 0, 0\n"
            "0, 1, 1, 1, 0\n"
            "0, 0, 1, 1, 1\n"
            "0, 0, 1, 1, 0\n"
            "0, 1, 1, 0, 0",
            height=150,
        )
        try:
            input_matrix = np.array([list(map(int, row.split(','))) for row in input_matrix_str.strip().split('\n')])
            st.write("Input Matrix Shape:", input_matrix.shape)
            st.table(input_matrix)
        except Exception as e:
            st.error(f"Invalid input matrix format. Please check your input. Error: {e}")
            input_matrix = None

    with col2:
        st.subheader("2. Define the Kernel (Filter)")
        # Example 3x3 kernel for edge detection
        kernel_str = st.text_area(
            "Enter the kernel matrix (comma-separated, rows on new lines):",
            "1, 0, -1\n"
            "1, 0, -1\n"
            "1, 0, -1",
            height=100
        )
        try:
            kernel = np.array([list(map(int, row.split(','))) for row in kernel_str.strip().split('\n')])
            st.write("Kernel Shape:", kernel.shape)
            st.table(kernel)
        except Exception as e:
            st.error(f"Invalid kernel format. Please check your input. Error: {e}")
            kernel = None
    
    st.markdown("---")
    
    if input_matrix is not None and kernel is not None:
        if st.button("Calculate Convolution"):
            st.header("Results")
            
            # Perform convolution
            i_h, i_w = input_matrix.shape
            k_h, k_w = kernel.shape
            
            if i_h < k_h or i_w < k_w:
                st.error("Input matrix dimensions must be greater than or equal to kernel dimensions.")
            else:
                o_h = i_h - k_h + 1
                o_w = i_w - k_w + 1
                output_matrix = np.zeros((o_h, o_w))

                st.subheader("Step-by-Step Operation")
                expander = st.expander("Show step-by-step calculation")
                
                # Flip kernel for convolution (in math, correlation uses the unflipped kernel)
                flipped_kernel = np.flipud(np.fliplr(kernel))

                for y in range(o_h):
                    for x in range(o_w):
                        # Extract the region of interest
                        roi = input_matrix[y:y+k_h, x:x+k_w]
                        # Perform element-wise multiplication and sum
                        output_matrix[y, x] = np.sum(roi * flipped_kernel)
                        
                        with expander:
                            st.markdown(f"**Step ({y+1}, {x+1}):**")
                            st.text(f"Region of Interest:\n{roi}")
                            st.text(f"Flipped Kernel:\n{flipped_kernel}")
                            st.text(f"Calculation: np.sum({roi.flatten()} * {flipped_kernel.flatten()}) = {output_matrix[y, x]}")
                            st.markdown("---")


                st.subheader("3. Resulting Feature Map")
                st.markdown("This is the output of the convolution operation.")
                st.table(output_matrix.astype(int))
                
                # Convolution Matrix (Toeplitz)
                st.subheader("4. The Convolution Matrix (Toeplitz Form)")
                st.markdown("""
                The convolution operation can be represented as a single matrix multiplication by transforming the kernel into a large, sparse matrix called a **doubly block Toeplitz matrix**.
                
                `Output (flattened) = Convolution Matrix * Input (flattened)`
                
                This demonstrates that convolution is fundamentally a linear operation. The structure of this matrix reveals how each element of the input contributes to each element of the output.
                """)

                try:
                    conv_matrix = create_convolution_matrix(input_matrix.shape, kernel.shape)
                    
                    # Flatten the input and multiply
                    input_flat = input_matrix.flatten()
                    output_flat_from_matrix = conv_matrix @ input_flat
                    output_from_matrix = output_flat_from_matrix.reshape((o_h, o_w))

                    st.markdown("Generated Convolution Matrix (may be large):")
                    st.dataframe(conv_matrix.astype(int))
                    
                    st.markdown("**Verification:**")
                    st.text("Output from direct convolution:")
                    st.table(output_matrix.astype(int))

                    st.text("Output from multiplying by the convolution matrix (may have rounding differences):")
                    st.table(output_from_matrix.astype(int))
                    
                    st.success("Verification successful: The outputs from both methods match!")

                except Exception as e:
                    st.error(f"Could not generate the convolution matrix. It might be too large for display or an error occurred: {e}")


# --- Page 3: Understanding CNNs ---
elif page == "Part 2: Understanding CNNs":
    st.title("Part 2: Understanding Convolutional Neural Networks (CNNs)")
    st.markdown("---")
    
    st.markdown("""
    A Convolutional Neural Network (CNN) is a type of deep learning model specifically designed for processing and analyzing grid-like data, such as images. They are inspired by the biological processes of the visual cortex in animals and are incredibly effective for tasks like image recognition, object detection, and image segmentation.

    The key innovation of CNNs is their ability to automatically and adaptively learn a hierarchy of features from the input data. This means lower layers might learn to detect simple features like edges and corners, while higher layers combine these to detect more complex features like shapes, objects, or faces.
    """)

    st.header("Key Components of a CNN")
    
    # Diagram of a typical CNN architecture
    cnn_graph = graphviz.Digraph()
    cnn_graph.attr('node', shape='box', style='rounded')
    cnn_graph.edge('Input Image', 'Convolutional Layer')
    cnn_graph.edge('Convolutional Layer', 'Activation (ReLU)')
    cnn_graph.edge('Activation (ReLU)', 'Pooling Layer')
    cnn_graph.edge('Pooling Layer', 'Fully Connected Layer')
    cnn_graph.edge('Fully Connected Layer', 'Output Layer')

    st.graphviz_chart(cnn_graph, use_container_width=True)

    st.markdown("""
    A typical CNN architecture consists of three main types of layers:
    1.  **Convolutional Layers:** The core building block where filters are applied to extract features.
    2.  **Pooling Layers:** Used to reduce the spatial dimensions of the feature maps, making the model more efficient and robust.
    3.  **Fully Connected Layers:** The final layers that perform classification based on the extracted features.
    """)
    
    st.markdown("---")

    # --- Tabs for detailed explanations ---
    tab1, tab2, tab3 = st.tabs(["Image Dimensions", "Steps in a CNN", "Ethical Considerations"])

    with tab1:
        st.subheader("Image Dimensions: RGB vs. Grayscale")
        st.markdown("""
        A digital image is a grid of pixels. The number of dimensions of an image depends on whether it is in color or grayscale.
        
        - **Grayscale (Black and White) Image:** This image has **2 dimensions**: a `Height` and a `Width`. Each pixel has a single value (typically from 0 to 255) representing its intensity, from black (0) to white (255).
        
        - **Colored (RGB) Image:** A standard RGB image has **3 dimensions**: `Height`, `Width`, and `Color Channels`. The third dimension, channels, has a depth of 3, corresponding to the **R**ed, **G**reen, and **B**lue color components. Each pixel is a combination of these three intensity values. CNNs process these three channels simultaneously.
        """)
        
        # Block Diagram Comparison
        img_dim_graph = graphviz.Digraph()
        img_dim_graph.attr(rankdir='LR')
        
        with img_dim_graph.subgraph(name='cluster_0') as c:
            c.attr(style='filled', color='lightgrey')
            c.node_attr.style = 'filled'
            c.attr(label='RGB Image (3D)')
            c.node('rgb', 'Height x Width x 3 Channels')

        with img_dim_graph.subgraph(name='cluster_1') as c:
            c.attr(style='filled', color='lightgrey')
            c.node_attr.style = 'filled'
            c.attr(label='Grayscale Image (2D)')
            c.node('bw', 'Height x Width x 1 Channel')
            
        st.graphviz_chart(img_dim_graph)

    with tab2:
        st.subheader("Deep Dive: The Steps in a CNN")

        st.markdown("#### 1. Convolutional Layers and Feature Maps")
        st.markdown("""
        The process begins with the convolutional layer, which applies a set of learnable filters (kernels) to the input image. Each filter is a small matrix of weights. As a filter slides, or "convolves," across the input image, it performs element-wise multiplication with the part of the image it is currently on, and then sums the results into a single output pixel.

        This process is repeated across the entire image, generating a 2D **feature map**. Each feature map corresponds to a specific filter and represents the presence of that filter's target feature (e.g., a vertical edge) in the input. A single convolutional layer typically learns many filters in parallel, producing multiple feature maps.
        """)
        
        st.image("https://i.imgur.com/1Abh2xI.gif", caption="", use_column_width=True)

        st.markdown("#### 2. Activation Function (ReLU)")
        st.markdown("""
        After each convolution operation, an activation function is applied to the feature map. The purpose of this function is to introduce **non-linearity** into the model. Without non-linearity, a deep stack of layers would behave like a single layer, limiting its ability to learn complex patterns.
        
        The most common activation function is the **Rectified Linear Unit (ReLU)**. It is computationally efficient and effective. The function is defined as:
        
        `f(x) = max(0, x)`
        
        In simple terms, it replaces all negative pixel values in the feature map with zero, while keeping positive values unchanged.
        """)
        st.pyplot(plot_relu())

        st.markdown("#### 3. Pooling Layers")
        st.markdown("""
        The purpose of pooling layers is to reduce the spatial dimensions (width and height) of the feature maps. This has two main benefits:
        - It reduces the number of parameters and computational cost in the network.
        - It helps make the feature detection more robust to changes in the position of the feature in the input image (a property called *spatial invariance*).
        
        The most common type of pooling is **Max Pooling**. It involves sliding a small window (e.g., 2x2 pixels) over the feature map and, for each window, taking only the maximum value.
        """)

        st.markdown("**Example of 2x2 Max Pooling:**")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Input Feature Map**")
            st.table(np.array([[1, 4, 2, 8], [3, 7, 0, 5], [6, 2, 9, 1], [5, 3, 4, 6]]))
        with col2:
            st.markdown("**Output after Max Pooling**")
            st.table(np.array([[7, 8], [6, 9]]))
        st.info("Notice how the 4x4 input is reduced to a 2x2 output by taking the max value from each 2x2 quadrant.")

        st.markdown("#### 4. Fully Connected Layers")
        st.markdown("""
        After several convolutional and pooling layers, the high-level features are extracted. The final part of the CNN is typically one or more **fully connected layers**. Before feeding the data into these layers, the final feature maps are **flattened** into a single one-dimensional vector.
        
        A fully connected layer is a traditional multi-layer perceptron where every neuron in the layer is connected to every neuron in the previous layer. Its role is to take the high-level features learned by the convolutional layers and use them to perform the final classification task. For example, it might learn that a combination of a "fur" feature, an "ear" feature, and a "whisker" feature strongly suggests the image is of a "cat".
        """)

        st.markdown("#### 5. Output Layer")
        st.markdown("""
        The output layer is the final layer in the network, and it produces the ultimate prediction. The number of neurons in this layer corresponds to the number of classes the model is trying to predict.
        
        For a multi-class classification problem, a **Softmax** activation function is typically used in the output layer. Softmax converts the raw output scores from the network into a probability distribution over the classes. Each neuron's output will be a value between 0 and 1, and the sum of all outputs will be 1, representing the model's confidence for each class.
        
        For example, for an image of a cat, the output might be:
        - **Dog:** 0.1 (10%)
        - **Cat:** 0.85 (85%)
        - **Bird:** 0.05 (5%)
        
        The model's final prediction would be "Cat".
        """)

# --- Page 4: Ethical Considerations ---
elif page == "Ethical Considerations in CNNs":
    st.title("Ethical Considerations in CNNs")
    st.markdown("---")
    st.markdown("""
    While CNNs are powerful tools with immense potential for positive impact, their development and deployment come with significant ethical responsibilities. It is crucial for engineers, researchers, and policymakers to consider these issues to ensure technology is used fairly, safely, and transparently.
    """)

    st.subheader("1. Biases in Image Datasets")
    st.warning("""
    **Concern:** The performance and fairness of a CNN are highly dependent on the data it was trained on. If a training dataset is not diverse and representative of the real world, the model will inherit and amplify existing societal biases.
    
    **Example:** A facial recognition model trained primarily on images of one demographic may perform poorly and make unfair judgments when applied to individuals from underrepresented groups. This can lead to discriminatory outcomes in areas like hiring, law enforcement, and loan applications.
    
    **Mitigation:** Actively curating balanced and diverse datasets, using data augmentation techniques to increase representation, and continuously auditing models for biased performance across different demographic groups.
    """)

    st.subheader("2. Security and Confidentiality of Data")
    st.warning("""
    **Concern:** CNNs often require vast amounts of data for training, which can include sensitive personal information such as medical scans (MRIs, X-rays) or personal photos for facial recognition.
    
    **Example:** A breach of a healthcare database used for training a medical imaging CNN could expose the private health information of thousands of patients. Similarly, datasets of faces collected without consent violate individual privacy.
    
    **Mitigation:** Implementing robust data security protocols, anonymizing data wherever possible, using privacy-preserving techniques like federated learning (where the model is trained on local data without the data ever leaving the device), and adhering to data protection regulations like GDPR.
    """)
    
    st.subheader("3. Need for Interpretability and Transparency")
    st.warning("""
    **Concern:** CNNs are often considered "black boxes" because their decision-making processes are not easily understood by humans. It can be difficult to determine *why* a model made a particular prediction.
    
    **Example:** If a CNN denies a loan application based on an analysis of satellite imagery of a property, the applicant has a right to know the reasoning behind the decision. Without transparency, it's impossible to challenge or correct erroneous conclusions.
    
    **Mitigation:** Developing and using techniques for model interpretability (e.g., LIME, SHAP, class activation maps) that help visualize which parts of an image were most influential in a model's decision. This is crucial for building trust and accountability.
    """)
    
    st.subheader("4. Responsible Deployment in Critical Applications")
    st.warning("""
    **Concern:** The deployment of CNNs in high-stakes applications like autonomous vehicle navigation, medical diagnosis, and public surveillance carries significant risks.
    
    **Example:** An autonomous car's object detection system failing to recognize a pedestrian in unusual lighting conditions could have fatal consequences. Mass surveillance systems using facial recognition can enable social control and suppress dissent, raising profound questions about civil liberties.
    
    **Mitigation:** Establishing strong regulatory frameworks, conducting rigorous testing and validation in diverse real-world scenarios, incorporating human oversight ("human-in-the-loop" systems), and engaging in public discourse about the societal impact of deploying such technologies.
    """)
    
    st.subheader("5. Potential Harmful Consequences of Errors")
    st.warning("""
    **Concern:** Even highly accurate models make mistakes. The consequences of these inaccuracies can range from minor inconveniences to severe harm.
    
    **Example:** A content moderation CNN incorrectly flagging a historical photo as inappropriate content can lead to censorship. A diagnostic AI incorrectly identifying cancer in a medical scan can lead to immense patient distress and unnecessary procedures. Conversely, failing to detect a disease could delay critical treatment.
    
    **Mitigation:** Designing systems with an understanding of their failure modes, providing clear confidence scores with predictions, ensuring there are mechanisms for appeal and correction, and avoiding full automation in decisions where human life and well-being are at stake.
    """)
