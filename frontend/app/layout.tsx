import "./globals.css";


export const metadata = {
title: "Python Dojo",
description: "Ask Python questions and get grounded answers",
};


export default function RootLayout({ children }: { children: React.ReactNode }) {
return (
<html lang="en" className="dark">
<body className="bg-neutral-950 text-neutral-100 min-h-screen">
{children}
</body>
</html>
);
}