import { useState } from "react";

export default function Suggestions() {
  // State for form data
  const [formData, setFormData] = useState({
    firstName: "",
    lastName: "",
    email: "",
    phoneNumber: "",
    suggestion: ""
  });

  // State for form submission status
  const [submitting, setSubmitting] = useState(false);
  const [submitStatus, setSubmitStatus] = useState(null);
  const [errors, setErrors] = useState({});

  // Handle form input changes
  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: value
    }));
  };

  // Validate form data
  const validateForm = () => {
    const newErrors = {};
    
    if (!formData.firstName.trim()) {
      newErrors.firstName = "First name is required";
    }
    
    if (!formData.lastName.trim()) {
      newErrors.lastName = "Last name is required";
    }
    
    if (!formData.email.trim()) {
      newErrors.email = "Email is required";
    } else if (!/^\S+@\S+\.\S+$/.test(formData.email)) {
      newErrors.email = "Invalid email format";
    }
    
    if (!formData.phoneNumber.trim()) {
      newErrors.phoneNumber = "Phone number is required";
    } else if (!/^\d{10,15}$/.test(formData.phoneNumber.replace(/[^0-9]/g, ''))) {
      newErrors.phoneNumber = "Phone number must be 10-15 digits";
    }
    
    if (!formData.suggestion.trim()) {
      newErrors.suggestion = "Suggestion is required";
    } else if (formData.suggestion.length < 10) {
      newErrors.suggestion = "Suggestion must be at least 10 characters";
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // Handle form submission
  const handleSubmit = async (e) => {
    e.preventDefault();
    
    // Validate form before submission
    if (!validateForm()) {
      return;
    }
    
    setSubmitting(true);
    setSubmitStatus(null);
    
    try {
      const response = await fetch("http://localhost:8000/api/suggestions/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          first_name: formData.firstName,
          last_name: formData.lastName,
          email: formData.email,
          phone_number: formData.phoneNumber,
          suggestion: formData.suggestion
        }),
      });
      
      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || "Failed to submit suggestion");
      }
      
      // Successfully submitted
      setSubmitStatus("success");
      // Reset form data
      setFormData({
        firstName: "",
        lastName: "",
        email: "",
        phoneNumber: "",
        suggestion: ""
      });
      
    } catch (error) {
      console.error("Error submitting form:", error);
      setSubmitStatus("error");
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="bg-stone-300 mt-20 min-h-screen w-screen" id="suggestions">
      <div className="">
        <h2 className="text-xl md:text-3xl lg:text-5xl text-black text-center" id="op3">
          Suggestions
        </h2>
        <p className="text-gray-800 font-md text-sm md:text-lg lg:text-xl text-center mt-2">
          Have a question or a suggestion? Reach out, and let's craft the perfect
          solution with our tools and services
        </p>
        <div className="flex flex-col lg:flex-row mx-auto lg:justify-start mt-20 md:space-x-10 lg:space-x-20 md:ml-10 lg:ml-30">
          <div className="flex flex-col">
            <p className="text-2xl text-black font-bold ml-20">
              Fill in the form below
            </p>
            <form onSubmit={handleSubmit}>
              <div className="flex space-x-4">
                <div className="flex flex-col">
                  <input
                    type="text"
                    name="firstName"
                    value={formData.firstName}
                    onChange={handleChange}
                    className={`rounded-lg px-5 w-42 md:w-68 py-3 bg-stone-100 mt-4 ml-20 shadow-xl text-lg hover:scale-110 ${
                      errors.firstName ? "border-2 border-red-500" : ""
                    }`}
                    placeholder="First Name"
                  />
                  {errors.firstName && (
                    <span className="text-red-500 text-sm ml-20 mt-1">{errors.firstName}</span>
                  )}
                </div>
                <div className="flex flex-col">
                  <input
                    type="text"
                    name="lastName"
                    value={formData.lastName}
                    onChange={handleChange}
                    className={`rounded-lg px-5 w-42 md:w-68 py-3 bg-stone-100 mt-4 shadow-xl text-lg hover:scale-110 ${
                      errors.lastName ? "border-2 border-red-500" : ""
                    }`}
                    placeholder="Last Name"
                  />
                  {errors.lastName && (
                    <span className="text-red-500 text-sm mt-1">{errors.lastName}</span>
                  )}
                </div>
              </div>
              <div className="flex flex-col">
                <input
                  type="email"
                  name="email"
                  value={formData.email}
                  onChange={handleChange}
                  className={`rounded-lg px-5 w-100 md:w-140 py-3 bg-stone-100 mt-8 ml-20 shadow-xl text-lg hover:scale-110 ${
                    errors.email ? "border-2 border-red-500" : ""
                  }`}
                  placeholder="Email"
                />
                {errors.email && (
                  <span className="text-red-500 text-sm ml-20 mt-1">{errors.email}</span>
                )}
              </div>
              <div className="flex flex-col">
                <input
                  type="tel"
                  name="phoneNumber"
                  value={formData.phoneNumber}
                  onChange={handleChange}
                  className={`rounded-lg px-5 w-100 md:w-140 py-3 bg-stone-100 mt-8 ml-20 shadow-xl text-lg hover:scale-110 ${
                    errors.phoneNumber ? "border-2 border-red-500" : ""
                  }`}
                  placeholder="Phone Number"
                />
                {errors.phoneNumber && (
                  <span className="text-red-500 text-sm ml-20 mt-1">{errors.phoneNumber}</span>
                )}
              </div>
              <div className="flex flex-col">
                <textarea
                  name="suggestion"
                  value={formData.suggestion}
                  onChange={handleChange}
                  className={`rounded-lg px-5 w-100 md:w-140 py-3 bg-stone-100 mt-8 h-40 ml-20 shadow-xl text-lg hover:scale-110 ${
                    errors.suggestion ? "border-2 border-red-500" : ""
                  }`}
                  placeholder="Suggestion"
                />
                {errors.suggestion && (
                  <span className="text-red-500 text-sm ml-20 mt-1">{errors.suggestion}</span>
                )}
              </div>
              <button
                type="submit"
                disabled={submitting}
                className="w-100 md:w-140 h-14 text-center text-white bg-gradient-to-r from-gray-600 via-gray-400 to-gray-400 rounded-lg ml-20 mt-8 text-xl font-bold shadow-xl transition-all duration-300 hover:cursor-pointer hover:scale-110"
              >
                {submitting ? "Submitting..." : "Submit"}
              </button>
              {submitStatus === "success" && (
                <p className="text-green-600 font-bold ml-20 mt-2">
                  Your suggestion has been submitted successfully!
                </p>
              )}
              {submitStatus === "error" && (
                <p className="text-red-600 font-bold ml-20 mt-2">
                  Failed to submit suggestion. Please try again.
                </p>
              )}
            </form>
            <p className="mt-4 mx-auto lg:ml-50 text-gray-600 font-lg">
              We'll get back to you in 1-2 business days.
            </p>
          </div>
          <div className="flex flex-col">
            <div className="flex space-x-3 mt-20">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="24"
                fill="currentColor"
                className="bi bi-question-circle ml-20 mt-2"
                viewBox="0 0 16 16"
              >
                <path d="M8 15A7 7 0 1 1 8 1a7 7 0 0 1 0 14m0 1A8 8 0 1 0 8 0a8 8 0 0 0 0 16" />
                <path d="M5.255 5.786a.237.237 0 0 0 .241.247h.825c.138 0 .248-.113.266-.25.09-.656.54-1.134 1.342-1.134.686 0 1.314.343 1.314 1.168 0 .635-.374.927-.965 1.371-.673.489-1.206 1.06-1.168 1.987l.003.217a.25.25 0 0 0 .25.246h.811a.25.25 0 0 0 .25-.25v-.105c0-.718.273-.927 1.01-1.486.609-.463 1.244-.977 1.244-2.056 0-1.511-1.276-2.241-2.673-2.241-1.267 0-2.655.59-2.75 2.286m1.557 5.763c0 .533.425.927 1.01.927.609 0 1.028-.394 1.028-.927 0-.552-.42-.94-1.029-.94-.584 0-1.009.388-1.009.94" />
              </svg>
              <div className="ml-10">
                <h2 className="text-xl text-black font-bold">Knowledgebase</h2>
                <p className="text-lg text-gray-800">
                  Browse through all of our Knowledgebase articles.
                </p>
                <div className="group">
                  <a
                    href="..."
                    className="text-lg text-gray-900 flex space-x-2 font-bold"
                  >
                    Visit our Database
                    <b>
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="24"
                        height="24"
                        fill="currentColor"
                        className="bi bi-arrow-right ml-1 mt-1 group-hover:scale-110 group-hover:translate-x-2 transition-all duration-300"
                        viewBox="0 0 16 16"
                      >
                        <path
                          fillRule="evenodd"
                          d="M1 8a.5.5 0 0 1 .5-.5h11.793l-3.147-3.146a.5.5 0 0 1 .708-.708l4 4a.5.5 0 0 1 0 .708l-4 4a.5.5 0 0 1-.708-.708L13.293 8.5H1.5A.5.5 0 0 1 1 8"
                        />
                      </svg>
                    </b>
                  </a>
                </div>
              </div>
            </div>
            <hr className="mt-8 text-stone-400 ml-20 max-w-200"></hr>
            <div className="flex space-x-3 mt-4">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="24"
                fill="currentColor"
                className="bi bi-chat-left ml-20 mt-2"
                viewBox="0 0 16 16"
              >
                <path d="M14 1a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1H4.414A2 2 0 0 0 3 11.586l-2 2V2a1 1 0 0 1 1-1zM2 0a2 2 0 0 0-2 2v12.793a.5.5 0 0 0 .854.353l2.853-2.853A1 1 0 0 1 4.414 12H14a2 2 0 0 0 2-2V2a2 2 0 0 0-2-2z" />
              </svg>
              <div className="ml-10">
                <h2 className="text-xl text-black font-bold">FAQ</h2>
                <p className="text-lg text-gray-800">
                  Explore our FAQ for quick, clear answers to common queries.
                </p>
                <div className="group">
                  <a
                    href="..."
                    className="text-lg text-gray-900 flex space-x-2 font-bold"
                  >
                    Visit FAQ
                    <b>
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="24"
                        height="24"
                        fill="currentColor"
                        className="bi bi-arrow-right ml-1 mt-1 group-hover:scale-110 group-hover:translate-x-2 transition-all duration-300"
                        viewBox="0 0 16 16"
                      >
                        <path
                          fillRule="evenodd"
                          d="M1 8a.5.5 0 0 1 .5-.5h11.793l-3.147-3.146a.5.5 0 0 1 .708-.708l4 4a.5.5 0 0 1 0 .708l-4 4a.5.5 0 0 1-.708-.708L13.293 8.5H1.5A.5.5 0 0 1 1 8"
                        />
                      </svg>
                    </b>
                  </a>
                </div>
              </div>
            </div>
            <hr className="mt-8 text-stone-400 ml-20 max-w-200"></hr>
            <div className="flex space-x-3 mt-4">
              <svg
                xmlns="http://www.w3.org/2000/svg"
                width="24"
                height="24"
                fill="currentColor"
                className="bi bi-envelope-open ml-20 mt-2"
                viewBox="0 0 16 16"
              >
                <path d="M8.47 1.318a1 1 0 0 0-.94 0l-6 3.2A1 1 0 0 0 1 5.4v.817l5.75 3.45L8 8.917l1.25.75L15 6.217V5.4a1 1 0 0 0-.53-.882zM15 7.383l-4.778 2.867L15 13.117zm-.035 6.88L8 10.082l-6.965 4.18A1 1 0 0 0 2 15h12a1 1 0 0 0 .965-.738ZM1 13.116l4.778-2.867L1 7.383v5.734ZM7.059.435a2 2 0 0 1 1.882 0l6 3.2A2 2 0 0 1 16 5.4V14a2 2 0 0 1-2 2H2a2 2 0 0 1-2-2V5.4a2 2 0 0 1 1.059-1.765z" />
              </svg>
              <div className="ml-10">
                <h2 className="text-xl text-black font-bold">
                  Contact us by email
                </h2>
                <p className="text-lg text-gray-800">
                  Prefer the written word? Drop us at email at{" "}
                </p>
                <div className="group">
                  <a
                    href="mailto:abcd@gmail.com"
                    className="text-lg text-gray-900 flex space-x-2 font-bold"
                  >
                    abcd@gmail.com
                    <b>
                      <svg
                        xmlns="http://www.w3.org/2000/svg"
                        width="24"
                        height="24"
                        fill="currentColor"
                        className="bi bi-arrow-right ml-1 mt-1 group-hover:scale-110 group-hover:translate-x-2 transition-all duration-300"
                        viewBox="0 0 16 16"
                      >
                        <path
                          fillRule="evenodd"
                          d="M1 8a.5.5 0 0 1 .5-.5h11.793l-3.147-3.146a.5.5 0 0 1 .708-.708l4 4a.5.5 0 0 1 0 .708l-4 4a.5.5 0 0 1-.708-.708L13.293 8.5H1.5A.5.5 0 0 1 1 8"
                        />
                      </svg>
                    </b>
                  </a>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}