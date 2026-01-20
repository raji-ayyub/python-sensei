import ChatBox from "@/components/ChatBox";
import logo from "@/app/assets/logo.png"
import Image from "next/image";


export default function Home() {
return (
<main className="flex items-center justify-center px-4 py-10">
<div className="w-full max-w-3xl">

<div className="flex gap-4 items-center my-4 ">
  <div className="w-[2.5rem] h-[2.5rem] bg-gray-100 rounded-md flex items-center justify-center">
    <Image src={logo} width={80} height={80} alt="logo" className="w-[2rem]"/>

  </div>

  <h1 className="text-3xl font-bold mb-2"> Python Sensei</h1>

</div>


<p className="text-neutral-400 mb-6">
Ask Python questions. Get clear, grounded answers.
</p>
<ChatBox />
</div>
</main>
);
}



















