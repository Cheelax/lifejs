"use client";
import { ArrowRight } from "lucide-react";
import { animate, stagger } from "motion";
import { motion } from "motion/react";
import { splitText } from "motion-plus";
import Image from "next/image";
import Link from "next/link";
import { useEffect, useRef, useState } from "react";
import { Button } from "@/components/ui/button";
import { FancyButton } from "@/components/ui/fancy-button";
import codeSnippet from "@/public/code-snippet.png";
import logoFull from "@/public/logo-full.png";
import onlydustLogo from "@/public/onlydust-logo-dark.png";
import { getGitHubStats } from "./actions/github-stats";

export default function Page() {
  const taglineRef = useRef<HTMLHeadingElement>(null);
  const descriptionRef = useRef<HTMLParagraphElement>(null);
  const [githubStats, setGithubStats] = useState({ stars: 3, forks: 2 });

  useEffect(() => {
    // Fetch GitHub stats
    const fetchGithubStats = async () => {
      try {
        const stats = await getGitHubStats();
        setGithubStats(stats);
      } catch (error) {
        console.error("Failed to fetch GitHub stats:", error);
        // Keep fallback values
      }
    };
    fetchGithubStats();

    document.fonts.ready.then(() => {
      // Animate tagline words
      if (taglineRef.current) {
        try {
          taglineRef.current.style.visibility = "visible";

          const { words } = splitText(taglineRef.current);

          // Apply gradient classes to each word to preserve the effect
          for (const word of words) {
            word.style.backgroundImage = "linear-gradient(to bottom, #222222, #444444)";
            word.style.backgroundClip = "text";
            word.style.webkitBackgroundClip = "text";
            word.style.color = "transparent";
            word.style.display = "inline-block";
          }

          animate(
            words,
            {
              opacity: [0, 1],
              y: [20, 0],
              filter: ["blur(8px)", "blur(0px)"],
            },
            {
              type: "spring",
              duration: 1.2,
              bounce: 0.1,
              delay: stagger(0.08, { startDelay: 0.2 }),
            },
          );
        } catch (error) {
          console.error("Tagline animation error:", error);
          // Fallback: just show the text
          if (taglineRef.current) {
            taglineRef.current.style.visibility = "visible";
            taglineRef.current.style.opacity = "1";
          }
        }
      }

      // Animate description words
      if (descriptionRef.current) {
        try {
          descriptionRef.current.style.visibility = "visible";

          const { words } = splitText(descriptionRef.current);

          animate(
            words,
            {
              opacity: [0, 1],
              y: [15, 0],
              filter: ["blur(6px)", "blur(0px)"],
            },
            {
              type: "spring",
              duration: 1,
              bounce: 0.05,
              delay: stagger(0.06, { startDelay: 0.8 }),
            },
          );
        } catch (error) {
          console.error("Description animation error:", error);
          // Fallback: just show the text
          if (descriptionRef.current) {
            descriptionRef.current.style.visibility = "visible";
            descriptionRef.current.style.opacity = "1";
          }
        }
      }
    });
  }, []);

  return (
    <div className="flex min-h-screen flex-col items-center">
      {/* Main Content */}
      <main className="flex w-full flex-col items-center gap-[3.75rem] px-[1.25rem] pt-[5rem] pb-[3.125rem]">
        <div className="flex flex-col items-center gap-[3.75rem]">
          <div className="flex flex-col items-center gap-[3.125rem]">
            {/* Logo */}
            <motion.div
              animate={{
                opacity: 0.6,
                y: 0,
                filter: "blur(0px)",
              }}
              initial={{
                opacity: 0,
                y: -50,
                filter: "blur(20px)",
              }}
              transition={{
                duration: 1.2,
                ease: [0.21, 1.02, 0.73, 1],
                delay: 0,
              }}
              whileHover={{
                opacity: 1,
              }}
            >
              <Image alt="Life.js" className="h-[20px] w-auto" height={100} src={logoFull} />
            </motion.div>

            {/* Hero Section */}
            <div className="flex flex-col items-center gap-[2.5rem]">
              <div className="flex flex-col items-center gap-[0.91rem]">
                <h1
                  className="max-w-[29rem] text-center font-heading font-medium text-[3.1rem] leading-[3.5625rem] leading-trim tracking-[-0.09075rem]"
                  ref={taglineRef}
                  style={{ visibility: "hidden" }}
                >
                  The framework to build agentic web apps.
                </h1>
              </div>
              <p
                className="text-center font-body font-normal text-[17px] text-black/40 leading-none tracking-normal"
                ref={descriptionRef}
                style={{ visibility: "hidden" }}
              >
                Open-source, minimal, extensible, typesafe, and fullstack.
              </p>
            </div>

            {/* Code Example */}
            <motion.div
              animate={{
                opacity: [0, 0.5, 1],
                y: 0,
                filter: "blur(0px)",
              }}
              initial={{
                opacity: 0,
                y: 40,
                filter: "blur(10px)",
              }}
              transition={{
                duration: 1.5,
                ease: [0.21, 1.02, 0.73, 1],
                delay: 1.5,
              }}
            >
              <Image
                alt="Code Example"
                className="-mt-29 -mb-34 pointer-events-none h-[450px] w-auto"
                height={600}
                src={codeSnippet}
              />
            </motion.div>

            {/* CTA Section */}
            <motion.div
              animate={{
                opacity: [0, 0.5, 1],
                y: 0,
                filter: "blur(0px)",
              }}
              className="flex flex-col items-center gap-[1.25rem]"
              initial={{
                opacity: 0,
                y: 40,
                filter: "blur(10px)",
              }}
              transition={{
                duration: 1.5,
                ease: [0.21, 1.02, 0.73, 1],
                delay: 1.5,
              }}
            >
              <div className="flex items-center gap-[1.25rem]">
                {/* GitHub Stars */}
                <Link href="https://github.com/lifejs/lifejs" target="_blank">
                  <Button className="gap-2" size="sm" variant={"outline"}>
                    <svg
                      className="opacity-45"
                      role="img"
                      viewBox="0 0 24 24"
                      xmlns="http://www.w3.org/2000/svg"
                    >
                      <title>GitHub</title>
                      <path d="M12 .297c-6.63 0-12 5.373-12 12 0 5.303 3.438 9.8 8.205 11.385.6.113.82-.258.82-.577 0-.285-.01-1.04-.015-2.04-3.338.724-4.042-1.61-4.042-1.61C4.422 18.07 3.633 17.7 3.633 17.7c-1.087-.744.084-.729.084-.729 1.205.084 1.838 1.236 1.838 1.236 1.07 1.835 2.809 1.305 3.495.998.108-.776.417-1.305.76-1.605-2.665-.3-5.466-1.332-5.466-5.93 0-1.31.465-2.38 1.235-3.22-.135-.303-.54-1.523.105-3.176 0 0 1.005-.322 3.3 1.23.96-.267 1.98-.399 3-.405 1.02.006 2.04.138 3 .405 2.28-1.552 3.285-1.23 3.285-1.23.645 1.653.24 2.873.12 3.176.765.84 1.23 1.91 1.23 3.22 0 4.61-2.805 5.625-5.475 5.92.42.36.81 1.096.81 2.22 0 1.606-.015 2.896-.015 3.286 0 .315.21.69.825.57C20.565 22.092 24 17.592 24 12.297c0-6.627-5.373-12-12-12" />
                    </svg>
                    {githubStats.stars}
                  </Button>
                </Link>

                {/* Get Started */}
                <Link href="/docs/start-here/installation">
                  <FancyButton className="gap-1" size="sm">
                    Get started <ArrowRight className="size-3.5" strokeWidth={2.5} />
                  </FancyButton>
                </Link>
              </div>

              {/* CLI Command */}
              <div className="font-code font-medium text-[#333333] text-[0.8125rem] leading-none leading-trim tracking-normal opacity-75">
                <span className="text-[#e77823]">~</span>
                <span> npx life init</span>
              </div>
            </motion.div>
          </div>

          {/* Divider */}
          <motion.div
            animate={{
              opacity: [0, 0.5, 1],
              filter: "blur(0px)",
            }}
            className="h-px w-[500px] bg-[#dddddd]"
            initial={{
              opacity: 0,
              filter: "blur(6px)",
            }}
            transition={{
              duration: 0.8,
              ease: [0.21, 1.02, 0.73, 1],
              delay: 2,
            }}
          />

          {/* Footer */}
          <motion.div
            animate={{
              opacity: [0, 0.5, 1],
              filter: "blur(0px)",
            }}
            className="flex flex-col items-center gap-[1.5rem]"
            initial={{
              opacity: 0,
              filter: "blur(8px)",
            }}
            transition={{
              duration: 1,
              ease: [0.21, 1.02, 0.73, 1],
              delay: 2,
            }}
          >
            <div className="flex items-center gap-[0.5rem]">
              <span className="text-[0.95rem] text-black/30 leading-none leading-trim tracking-normal">
                Sponsored by
              </span>
              <Link href="https://onlydust.com" target="_blank">
                <Image
                  alt="OnlyDust"
                  className="h-[23px] w-auto opacity-28 transition-opacity duration-300 hover:opacity-38"
                  height={100}
                  src={onlydustLogo}
                />
              </Link>
            </div>
            <p className="font-handwriting text-[0.93rem] text-black/26 leading-none leading-trim tracking-normal">
              Made with love in SF.
            </p>
          </motion.div>
        </div>
      </main>

      {/* Styles for split text animation */}
      <style jsx>{`
        .split-word {
          will-change: transform, opacity;
        }
      `}</style>
    </div>
  );
}
