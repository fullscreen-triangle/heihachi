import { motion } from 'framer-motion'
import Link from 'next/link'
import React from 'react'


let MotionLink = motion(Link);

const Logo = () => {

  return (
    <div
     className='flex flex-col items-center justify-center mt-2'>
        <MotionLink href="/"
    className='flex items-center justify-center rounded-full w-16 h-16 bg-dark text-white dark:border-2 dark:border-solid dark:border-light
    text-lg font-bold tracking-tight'
    whileHover={{
      backgroundColor:["#121212", "#0ea5e9","#8b5cf6","#06b6d4","#0ea5e9", "#121212"],
      transition:{duration:1, repeat: Infinity }
    }}
    >YKZ</MotionLink>
    </div>
  )
}

export default Logo
