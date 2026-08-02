function validateForm() {
    let name = document.getElementById("name").value;
    let message = document.getElementById("message").value;

    if (name == "" || message == "") {
        alert("Please fill all fields");
        return false;
    }

    return true;
}