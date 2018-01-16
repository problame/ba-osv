package main

import (
    "errors"
    "io"
    "net"
    "log"
    "time"
    "os"
    "strconv"
    "github.com/kr/pretty"
)

type result struct {
    T time.Duration
    err error
}

func conn(remote string, res chan result) {

    to := 1*time.Second

    dialer := net.Dialer{
        Timeout: to,
    }
    conn, err := dialer.Dial("tcp", remote)
    if err != nil {
        if neterr, ok := err.(net.Error); ok && neterr.Timeout() {
            log.Print("timeout dialing")
            return
        } else {
            log.Panic(err)
        }
    }

    var buf [16]byte

    start := time.Now()
    conn.SetReadDeadline(start.Add(to))
    n, err := conn.Read(buf[:])
    end := time.Now()
    conn.Close()

    if err != nil && err != io.EOF {
        res <- result{0, err}
        log.Println(err)
    } else if n != 0 {
        res <- result{0, errors.New("weird benchmark")}
    } else {
        // this was probably ok
        res <- result{end.Sub(start), nil}
    }
}

func main() {

    remote := os.Args[1]
    benchTimerTime, err := time.ParseDuration(os.Args[2])
    if err != nil {
        log.Panic(err)
    }
    parallelClients, err := strconv.Atoi(os.Args[3])
    if err != nil {
        log.Panic(err)
    }

    rchan := make(chan result, 1024)
    for c := 0; c < parallelClients; c++ {
        go func() {
            for {
                conn(remote, rchan)
            }
        }()
    }

    durations := make([]time.Duration, 0)
    var errors int64

    timer := time.NewTimer(benchTimerTime)

    benchStart := time.Now()
    outer:
    for {
        select {
        case <-timer.C:
            break outer
        case res := <-rchan:
            if res.err != nil {
                errors += 1
            } else {
                durations = append(durations, res.T)
            }
        }
    }
    benchEnd := time.Now()


    // statistics
    type Stats struct {
        runTime time.Duration
        avgDuration float64
        avgConnsSec float64
        avgErrorsSec float64
    }
    var stats Stats
    stats.runTime = benchEnd.Sub(benchStart)

    var totalDuration time.Duration
    for _, d := range durations {
        totalDuration += d
    }
    stats.avgDuration = totalDuration.Seconds() / float64(len(durations))

    stats.avgConnsSec = float64(len(durations)) / stats.runTime.Seconds()
    stats.avgErrorsSec = float64(errors) / stats.runTime.Seconds()

    pretty.Println(stats)

}
