
TARGET=surveillance
SRCS=main.cpp SurveillanceCamera.cpp
OBJS=$(SRCS:.cpp=.o)

CC=g++
CFLAGS=-O2 -g -std=c++14
INCDIR=-I/usr/include/opencv4
LIBDIR=
LIBS=-lopencv_core -lopencv_dnn -lopencv_imgcodecs -lopencv_imgproc -lopencv_objdetect -lopencv_videoio -lopencv_video -lpthread


$(TARGET): $(OBJS)
	$(CC) -o $@ $^ $(LIBDIR) $(LIBS)

$(OBJS): $(SRCS)
	$(CC) $(CFLAGS) $(INCDIR) -c $(SRCS) 

all: clean $(OBJS) $(TARGET)
	./surveillance

clean:
	rm -f $(OBJS) $(TARGET) *.d
