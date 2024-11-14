document.getElementById("uploadBtn").addEventListener("click", async () => {
  const fileInput = document.getElementById("fileInput");
  if (fileInput.files.length === 0) {
    alert("Please select a file.");
    return;
  }

  const formData = new FormData();
  formData.append("file", fileInput.files[0]);

  try {
    const response = await fetch("http://127.0.0.1:5000/predict", {
      method: "POST",
      body: formData,
    });

    if (response.ok) {
      const data = await response.json();
      // Assuming the server returns the filename of the saved image
      const img = document.createElement("img");
      img.src = data["image_url"]; // Adjust the path based on your server setup
      document.getElementById("result").innerHTML = ""; // Clear previous results
      document.getElementById("result").appendChild(img);
    } else {
      const data = await response.json();
      alert(data.error || "Error processing the file.");
    }
  } catch (error) {
    console.error("Error:", error);
    alert("Error processing the file.");
  }
});
