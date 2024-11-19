package main

import (
	"fmt"
	"io"
	"net"
	"net/rpc"
	"os"
)

type Server struct{}

func (this *Server) SendFile(filename string, reply *[]byte) error {
	// Ensure the file exists in the current directory
	file, err := os.Open(filename)
	if err != nil {
		return err
	}
	defer file.Close()

	// Read the file content
	content, err := io.ReadAll(file)
	if err != nil {
		return err
	}

	// Set the reply to the file content
	*reply = content
	return nil
}

func server() {
	rpc.Register(new(Server))
	ln, err := net.Listen("tcp", "0.0.0.0:8080")
	if err != nil {
		fmt.Println(err)
		return
	}
	for {
		c, err := ln.Accept()
		if err != nil {
			continue
		}
		go rpc.ServeConn(c)
	}
}

func main() {
	server()
	// go client()
	var input string
	fmt.Scanln(&input)
}
