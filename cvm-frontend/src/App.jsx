import { Routes, Route } from "react-router-dom"
import Hero from "./components/Hero"
import ValueSection from "./components/ValueSection"
import HowItWorks from "./components/HowItWorks"
import Analysis from "./components/Analysis"

function Home() {
  return (
    <>
      <Hero />
      <ValueSection />
      <HowItWorks />
    </>
  )
}

function App() {
  return (
    <Routes>
      <Route path="/"         element={<Home />} />
      <Route path="/analysis" element={<Analysis />} />
    </Routes>
  )
}

export default App
