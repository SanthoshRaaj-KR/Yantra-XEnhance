function handleClick(){
    window.location.href="#Enhance";
  }
  
  export default function Hero(){
      return(
          <div class="flex  mr-22  justify-end  ">
              <div class="text-center">
                <p onClick={()=>{console.log("iauhfh")}} class="text-5xl text-white font-md  mt-48">We Don't Enhance <b class="text-5xl  font-medium text-red-500">X-Rays</b></p>
                <p class="text-5xl text-white font-md  mt-2">We Enhance <b class="text-5xl text-red-500 font-medium">Lives</b></p>
                <p class="text-white text-md  mt-8 px-4 text-3xl font-light">Empowering the world with advanced Radiology AI.</p>
                <p class="text-white text-md   px-4 text-3xl font-light">Making the worst, better And  better, the best.</p>
                <p class="text-white text-md   px-4 text-3xl font-light">The best enhancing AI there is</p>
                <button class="rounded-full px-12 py-4 border-2 transition-all bg-gradient-to-r from-stone-100 via-stone-400 to-stone-500 font-bold hover:bg-gradient-to-r hover:from-stone-500 hover:via-stone-400 hover:to-black  hover:text-white mb-2 mt-30  duration-300 hover:scale-90 hover:cursor-pointer" onClick={handleClick} >ENHANCE</button>
  
              </div>
            </div>
      )
  }