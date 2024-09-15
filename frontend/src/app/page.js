"use client"
import dynamic from "next/dynamic";
import { data } from "./data";

const ForceGraph3DComponent = dynamic(() => import('./components/ForceGraph3DComponent'), {
  ssr: false,
  loading: () => <p>Loading 3D graph...</p>
});

export default function Home() {
  return (
    <div className="app">
      <ForceGraph3DComponent />
    </div>
  );
}
