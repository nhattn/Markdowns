---
title: Docker multi-stage build with example
link: https://programmerlib.com/docker-multi-stage-build-with-example/
author: Adelaide
---

![](https://programmerlib.com/wp-content/uploads/2020/12/docker-multi-stage-build.png)

Today, let us look at the multi-stage build of docker and what problems it
can solve.

With the release of version `17.05`, Docker has also made an important update
to the image building, which is **multi-stage build**. This is really helpful
for the developers who have long been troubled by the image size.

Before version `17.05`, when we built Docker images, we may face some troubles:

1. Writing all the build processes in the one Dockerfile, including the compiling,
testing, and packaging libraries of the project and it would result in
some problems: 
   - Dockerfile is particularly bloated and poorly readable
   - The image level is very deep
   - A risk of source code leakage
2. A better way is to package the project and its dependent libraries externally
beforehand, and then copy it to the build directory. Although this can well
avoid the risk of the first method, it still needs to write two Dockerfile
and some scripts to automatically integrate the two stages.

Here is a simple Java project as an example to show the process of building
and this project only contains an `Demo.java` class:

```java
public class Demo {
    public static void main(String[] args) {
        System.out.println("Hello, multi-stage build");
    }
}

```

Now let's look at how to write dockerfile based on this project：

```dockerfile
FROM dojomadness/alpine-jdk8-maven:latest

# add pom.xml and source code
ADD ./pom.xml pom.xml
ADD ./src src/

# package jar and remove source code and temporary class files
RUN mvn clean package && cp -f target/msb-1.0.jar msb.jar && rm -rf pom.xml src/ target/

# run jar
CMD ["java", "-jar", "msb.jar"]

```

Here, we write all the build processes in the same Dockerfile, but in order
to reduce the image level as much as possible, we merge multiple execution
commands into the same `RUN` instruction. At the same time, we have to clean
up the source code file and temporary directory file and compile by ourselves.

Nevertheless, the container size built by this dockerfile will still be
relatively large, so we have to continue to improve：

Here we divide the process into two stages:

**Step 1**: Create first dockerfile used to compile source code:

```dockerfile
FROM maven:3.5.0-jdk-8-alpine

# add pom.xml and source code
ADD ./pom.xml pom.xml
ADD ./src src/

# package jar
RUN mvn clean package

```

**Step 2**: Create second dockerfile as run environment:

```dockerfile
From openjdk:8-jre-alpine

# copy jar
COPY ./msb.jar msb.jar

# run jar
CMD ["java", "-jar", "msb.jar"]

```

**Step 3**: Create a shell script to integrate above two dockerfile:

```bash
#!/bin/bash

# First stage: complete build environment
docker build -t msb:build -f Dockerfile.2.1 .
# create temporary container
docker create --name extract msb:build
# extract jar
docker cp extract:/target/msb-1.0.jar ./msb.jar
# remove temporary container
docker rm -f extract

# Second stage: minimal runtime environment
docker build --no-cache -t msb:build-2 -f Dockerfile.2.2 .

# remove local jar
rm -rf ./msb.jar

```

Here we have written two Dockerfiles and a Shell script to build the final
image in two stages, but if our projects are related to each other, then
we need to maintain multiple dockerfiles in the shell script. If there are
more projects, it will be troublesome to maintain.

Still the above example, let us see how to achieve it with multi-stage build:

```dockerfile
# First stage: complete build environment
FROM maven:3.5.0-jdk-8-alpine AS builder
# add pom.xml and source code
ADD ./pom.xml pom.xml
ADD ./src src/
# package jar
RUN mvn clean package

# Second stage: minimal runtime environment
From openjdk:8-jre-alpine
# copy jar from the first stage
COPY --from=builder target/msb-1.0.jar msb.jar
# run jar
CMD ["java", "-jar", "msb.jar"]

```

The above Dockerfile just merges the two Dockerfiles into the one Dockerfile,
and automatically completes the script in step3 process for us during the
build process.

For multi-stage build, there are two key points:

1.  An `AS` parameter is added after the FROM instruction in the previous stage,
which can name the stage for easy reference in subsequent stages. The format
is as follows:

    ```dockerfile
    FROM image[:tag | @digest] AS stage
    ```
2. The `--from` parameter is added after the COPY instruction in the subsequent
phases, which is used to specify which results of the previous phase to refer
to. Here is the format:

    ```dockerfile
    COPY --from=stage ...
    ```

The **multi-stage build** can also easily build the container image from
multiple dependent projects through one Dockerfile without worrying about
the large size of an image and source code leakage.
