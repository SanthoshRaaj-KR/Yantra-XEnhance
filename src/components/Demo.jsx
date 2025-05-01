import Img2 from '../assets/HeadAfter.jpg';
import Img1 from '../assets/HeadBefore.jpg';
import Img3 from '../assets/LegBefore.jpg';
import Img4 from '../assets/LegAfter.jpg';

export default function Demo() {
  return (
    <div class="mt-44 w-full h-screen  bg-stone-300 " id="demo">
      <div class=" flex justify-center  ">
        <p class=" text-6xl  mx-auto mt-20">Demo</p>
      </div>
      <div class=" lg:ml-30 flex space-x-4 justify-center lg:space-x-0 lg:block mt-10 lg:mt-0">
        <p class="text-black text-xl font-bold absolute max-w-50 lg:mt-10 mb-2 hidden lg:block">
          As you can see the X-ray before is full of noises and disturbances.
        </p>
          <div class="flex flex-col lg:block">
        <p class="lg:absolute lg:mt-70 lg:ml-32 text-black text-2xl">Before</p>
        <div class="lg:absolute lg:mt-80 lg:ml-20 bg-stone-100  h-32 w-32 md:w-64 md:h-64 lg:w-96 lg:h-96 rounded-lg text-black text-md font-light shadow-2xl shadow-stone-600 transition-all duration-200 hover:w-180 hover:h-160 hover:mt-20 hover:z-10 ">
        <img src={Img1} alt="HeadAfter" class="h-32 w-32 md:w-64 md:h-64 lg:w-96 lg:h-96 transition-all duration-200 hover:w-180 hover:h-160  hover:z-10 rounded-lg"/>
        </div>
        <p class="lg:absolute lg:mt-10 lg:ml-90 text-black text-2xl">After</p>
        <div class="lg:absolute lg:mt-20 lg:ml-80 bg-stone-500 h-32 w-32 md:w-64 md:h-64 lg:w-96 lg:h-96 rounded-lg text-white text-md font-light shadow-2xl shadow-stone-600 transition-all duration-200 hover:w-180 hover:h-160 hover:mt-20 hover:z-10">
          <img src={Img2} alt="HeadAfter" class="h-32 w-32 md:w-64 md:h-64 lg:w-96 lg:h-96 transition-all duration-200 hover:w-180 hover:h-160  hover:z-10 rounded-lg"/>
        </div>
        </div>
        <div class="flex flex-col lg:block">
        <p class="lg:absolute lg:mt-70 lg:ml-180 text-black text-2xl">Before</p>
        <div class="lg:absolute lg:mt-80 lg:ml-160 bg-stone-100 h-32 w-32 md:w-64 md:h-64 lg:w-96 lg:h-96 rounded-lg text-black text-md font-light shadow-2xl shadow-stone-600 transition-all duration-200 hover:w-180 hover:h-160 hover:mt-20 hover:z-10">
        <img src={Img3} alt="HeadAfter" class=" h-32 w-32 md:w-64 md:h-64 lg:w-96 lg:h-96 transition-all duration-200 hover:w-180 hover:h-160  hover:z-10 rounded-lg"/>
        </div>
        <p class="lg:absolute lg:mt-10 lg:ml-254 text-black text-2xl">After</p>
        <div class="lg:absolute lg:mt-20 lg:ml-240 bg-stone-500 h-32 w-32 md:w-64 md:h-64 lg:w-96 lg:h-96 rounded-lg text-white text-md font-light shadow-2xl shadow-stone-600 transition-all duration-200 hover:w-180 hover:h-160 hover:mt-20 hover:z-10 hover:ml-180">
        <img src={Img4} alt="HeadAfter" class=" h-32 w-32 md:w-64 md:h-64 lg:w-96 lg:h-96 transition-all duration-200 hover:w-180 hover:h-160  hover:z-10 rounded-lg"/>
        </div>
        </div>
        <p class="text-black text-xl font-bold absolute max-w-60 lg:ml-300 lg:mt-130 hidden lg:block">
          After enhancing them you can see the x-rays are far better , cleaner
          and noice is much less.
        </p>
      </div>
    </div>
  );
}