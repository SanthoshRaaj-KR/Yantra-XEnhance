// src/components/Demo.jsx

import Img2 from '../assets/HeadAfter.jpg';
import Img1 from '../assets/HeadBefore.jpg';
import Img3 from '../assets/LegBefore.jpg';
import Img4 from '../assets/LegAfter.jpg';

// Step 1: Create a reusable component for the "Before" and "After" image pairs.
// This makes the code cleaner and easier to manage.
function ImagePair({ beforeImg, afterImg }) {
  return (
    <div className="flex flex-col md:flex-row gap-4 md:gap-8 items-center justify-center">
      {/* Before Image */}
      <div className="text-center">
        <p className="text-2xl font-semibold mb-2">Before</p>
        <img
          src={beforeImg}
          alt="Before"
          className="w-48 h-48 md:w-64 md:h-64 lg:w-96 lg:h-96 object-cover rounded-lg shadow-lg transition-transform duration-300 hover:scale-105"
        />
      </div>

      {/* After Image */}
      <div className="text-center">
        <p className="text-2xl font-semibold mb-2">After</p>
        <img
          src={afterImg}
          alt="After"
          className="w-48 h-48 md:w-64 md:h-64 lg:w-96 lg:h-96 object-cover rounded-lg shadow-lg transition-transform duration-300 hover:scale-105"
        />
      </div>
    </div>
  );
}


export default function Demo() {
  return (
    // Use padding instead of margin-top for better spacing control
    <div className="w-full min-h-screen bg-stone-300 py-24 px-4" id="demo">
      {/* Main container to center content with a max-width */}
      <div className="max-w-7xl mx-auto flex flex-col items-center text-center">
        
        <h1 className="text-6xl font-bold mb-8">Demo</h1>

        <p className="text-xl font-semibold text-stone-800 max-w-2xl mb-12">
          A clear depiction of how our AI model enhances noisy X-ray images, making them far clearer and easier to analyze.
        </p>

        {/* Use a flex container for the image pairs */}
        <div className="flex flex-col gap-16 lg:gap-8">
          {/* First Pair (Head) */}
          <ImagePair beforeImg={Img1} afterImg={Img2} />

          {/* Second Pair (Leg) */}
          <ImagePair beforeImg={Img3} afterImg={Img4} />
        </div>

      </div>
    </div>
  );
}